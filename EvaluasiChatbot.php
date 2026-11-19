<?php

namespace App\Console\Commands;

use App\Http\Controllers\ChatbotController;
use App\Models\Pengguna;
use App\Support\Penjaga;
use App\Support\Pencari;
use Illuminate\Console\Command;
use Illuminate\Http\Client\Events\ResponseReceived;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Event;
use Throwable;

/**
 * Pengujian RAGAS: menjalankan setiap pertanyaan pada dataset melalui ChatbotController::tanya (alur produksi),
 * lalu menyimpan jawaban, potongan konteks (Pencari::cari), waktu respons LLM, dan jumlah token ke JSON.
 */
class EvaluasiChatbot extends Command
{
    protected $signature = 'chatbot:evaluasi {dataset : CSV dataset uji} {keluaran=hasil_chatbot.json : berkas JSON hasil}
        {--jeda=2 : detik antar pertanyaan (batas laju API)}
        {--id= : hanya jalankan id ini (pisah koma), hasil digabung ke berkas keluaran yang sudah ada}
        {--model= : ganti model LLM penyusun jawaban}
        {--url= : ganti alamat API chat completions}
        {--kunci= : nama variabel .env berisi API key untuk model ini, mis. GEMINI_API_KEY}';

    protected $description = 'Ekspor jawaban, konteks, waktu, dan token chatbot untuk evaluasi RAGAS';

    /** @var array{waktu_ms: ?int, token_masuk: ?int, token_keluar: ?int} */
    private array $llm = ['waktu_ms' => null, 'token_masuk' => null, 'token_keluar' => null];

    public function handle(): int
    {
        $baris = $this->bacaCsv($this->jalur($this->argument('dataset')));
        if ($baris === []) {
            $this->error('Dataset kosong atau tidak terbaca.');

            return self::FAILURE;
        }

        if ($this->option('model')) {
            config(['services.groq.model' => $this->option('model')]);
        }
        if ($this->option('url')) {
            config(['services.groq.url' => $this->option('url')]);
        }
        if ($this->option('kunci')) {
            $kunci = env($this->option('kunci'));
            if (! $kunci) {
                $this->error('Variabel '.$this->option('kunci').' kosong di .env.');

                return self::FAILURE;
            }
            config(['services.groq.key' => $kunci]);
        }
        $model = config('services.groq.model');
        $this->info("Model: {$model} | URL: ".config('services.groq.url'));

        // Catat waktu respons dan token dari panggilan ke LLM (bukan ke Gemini embedding).
        Event::listen(ResponseReceived::class, function (ResponseReceived $e) {
            if (! str_starts_with((string) $e->request->url(), (string) config('services.groq.url'))) {
                return;
            }
            $this->llm = [
                'waktu_ms' => (int) round(($e->response->transferStats?->getTransferTime() ?? 0) * 1000),
                'token_masuk' => $e->response->json('usage.prompt_tokens'),
                'token_keluar' => $e->response->json('usage.completion_tokens'),
            ];
        });

        Cache::flush();
        $hasil = [];
        $hanya = array_filter(array_map('trim', explode(',', (string) $this->option('id'))));
        if ($hanya) {
            $baris = array_values(array_filter($baris, fn ($r) => in_array($r['id'], $hanya, true)));
        }
        $bar = $this->output->createProgressBar(count($baris));

        foreach ($baris as $r) {
            $pengguna = Pengguna::where('peran', $r['peran'])->first();
            if (! $pengguna) {
                $this->warn("Tidak ada pengguna berperan {$r['peran']}; baris {$r['id']} dilewati.");
                continue;
            }
            $req = Request::create('/api/chatbot/tanya', 'POST', ['pertanyaan' => $r['pertanyaan']]);
            $req->setUserResolver(fn () => $pengguna);

            // Ulangi sampai 3 kali bila API sibuk (batas laju); error tetap dicatat sebagai [ERROR].
            for ($coba = 1; $coba <= 3; $coba++) {
                $konteks = [];
                $this->llm = ['waktu_ms' => null, 'token_masuk' => null, 'token_keluar' => null];
                $mulai = microtime(true);
                try {
                    $konteks = Pencari::cari($r['pertanyaan'])->map(fn ($x) => Penjaga::bersihkan($x['potongan']->isi))->all();
                    $mulai = microtime(true); // waktu total tanya(): embedding pertanyaan + pencarian + LLM
                    $resp = app(ChatbotController::class)->tanya($req);
                    $isi = $resp->getData(true);
                    $jawaban = $isi['jawaban'] ?? ('[ERROR] '.($isi['message'] ?? ''));
                    $luar = (bool) ($isi['di_luar_cakupan'] ?? false);
                    $salahPeran = (bool) ($isi['salah_peran'] ?? false);
                } catch (Throwable $e) {
                    $jawaban = '[ERROR] '.$e->getMessage();
                    $luar = false;
                    $salahPeran = false;
                }
                $total = (int) round((microtime(true) - $mulai) * 1000);
                if (! str_starts_with($jawaban, '[ERROR]')) {
                    break;
                }
                sleep(15 * $coba);
            }

            $hasil[] = $r + [
                'model' => $model,
                'jawaban_chatbot' => $jawaban,
                'di_luar_cakupan' => $luar,
                'salah_peran' => $salahPeran,
                'konteks' => $konteks,
                'waktu_total_ms' => $total,
            ] + $this->llm;
            $bar->advance();
            sleep((int) $this->option('jeda'));
        }
        $bar->finish();
        $keluaran = $this->jalur($this->argument('keluaran'));
        if ($hanya && is_file($keluaran)) { 
            $lama = collect(json_decode(file_get_contents($keluaran), true) ?: [])->keyBy('id');
            foreach ($hasil as $h) {
                $lama[$h['id']] = $h;
            }
            $hasil = $lama->sortBy(fn ($h) => (int) $h['id'])->values()->all();
        }
        file_put_contents($keluaran, json_encode($hasil, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));
        $this->newLine();
        $this->info(count($hasil).' pertanyaan disimpan ke '.$keluaran);

        return self::SUCCESS;
    }

   
    private function jalur(string $path): string
    {
        if (is_file($path) || preg_match('~^([A-Za-z]:)?[\\\\/]~', $path) === 1) {
            return $path;
        }
        $dir = dirname(base_path()).DIRECTORY_SEPARATOR.'ragas-eval';

        return is_dir($dir) ? $dir.DIRECTORY_SEPARATOR.$path : $path;
    }

    private function bacaCsv(string $path): array
    {
        if (! is_file($path) || ($h = fopen($path, 'r')) === false) {
            return [];
        }
        $kepala = array_map(fn ($k) => trim($k, "\xEF\xBB\xBF \t"), fgetcsv($h) ?: []);
        $baris = [];
        while (($c = fgetcsv($h)) !== false) {
            if (count($c) === count($kepala)) {
                $baris[] = array_combine($kepala, $c);
            }
        }
        fclose($h);

        return $baris;
    }
}
