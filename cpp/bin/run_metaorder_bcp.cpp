#include <iostream>
#include <chrono>
#include <future>
#include <vector>
#include <filesystem>
#include <thread>
#include <mutex>
#include <fstream>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"

using namespace qr;

struct Accumulator {
    std::vector<int64_t> grid;
    std::vector<double> mid_sum;
    std::vector<double> bias_sum;
    std::vector<double> sign_mean_sum;
    std::mutex mtx;
    int count = 0;

    Accumulator(int64_t duration, int64_t step) {
        for (int64_t t = 0; t <= duration; t += step) {
            grid.push_back(t);

            mid_sum.push_back(0.0);
            bias_sum.push_back(0.0);
            sign_mean_sum.push_back(0.0);
        }
    }

    void add(const Buffer& buffer) {
        if (buffer.records.empty()) return;

        double mid0 = (buffer.records[0].best_bid_price +
                       buffer.records[0].best_ask_price) / 2.0;

        std::vector<double> proj_mid(grid.size(), 0.0);
        std::vector<double> proj_bias(grid.size(), 0.0);
        std::vector<double> proj_sign_mean(grid.size(), 0.0);

        size_t j = 0;
        for (size_t i = 0; i < grid.size(); i++) {
            while (j + 1 < buffer.records.size() &&
                   buffer.records[j + 1].timestamp <= grid[i]) {
                j++;
            }
            if (j < buffer.records.size()) {
                double mid = (buffer.records[j].best_bid_price +
                              buffer.records[j].best_ask_price) / 2.0;
                proj_mid[i] = mid - mid0;
                proj_bias[i] = buffer.records[j].bias;
                proj_sign_mean[i] = buffer.records[j].trade_sign_mean;
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < grid.size(); i++) {
            mid_sum[i] += proj_mid[i];
            bias_sum[i] += proj_bias[i];
            sign_mean_sum[i] += proj_sign_mean[i];
        }
        count++;
    }

    void save_csv(const std::string& path) {
        std::ofstream file(path);
        file << "timestamp,avg_mid_price_change,avg_bias,avg_trade_sign_mean\n";
        for (size_t i = 0; i < grid.size(); i++) {
            file << grid[i] << "," << (mid_sum[i] / count) << "," << (bias_sum[i] / count) << "," << (sign_mean_sum[i] / count) << "\n";
        }
    }
};

void run_and_accumulate(const std::string& data_path, const QueueDistributions& dists,
                        uint64_t seed, int64_t duration, Accumulator& acc) {
    OrderBook lob(dists, 4, seed);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(data_path);
    QRModel model(&lob, params, seed);

    // PowerLawImpact
    // double alpha, A, m, eps;
    // PowerLawImpact impact(alpha, A, m, eps);

    // LinearImpact: A = -B / (5 * t0) so impact reaches 0 at t0 + 5*t0
    // double B = 0.4;
    // int64_t t0 = 120LL * 1'000'000'000;
    // double A = -B / (5.0 * static_cast<double>(t0));
    // LinearImpact impact(B, A, t0);

    // NoImpact
    NoImpact impact;

    MetaOrder metaorder;
    metaorder.side = Side::Ask;
    for (int t = 0; t <= 120; t += 20) {
        metaorder.timestamps.push_back(static_cast<int64_t>(t) * 1'000'000'000);
        metaorder.sizes.push_back(3);
    }

    Buffer result = run_metaorder(lob, model, impact, metaorder, duration);
    acc.add(result);
}

int main(int argc, char* argv[]) {
    auto print_help = [&]() {
        std::cout << "Usage: " << argv[0] << " <ticker>\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help    Show this help message\n";
    };

    if (argc < 2) {
        print_help();
        return 1;
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_help();
        return 0;
    }

    std::string base_path = "/home/labcmap/saad.souilmi/dev_cpp/qr/data";
    std::string ticker = argv[1];
    std::string data_path = base_path + "/" + ticker;
    std::string base_results_path = base_path + "/results/" + ticker;

    std::cout << "Using ticker: " << ticker << "\n";

    // Load queue distributions once
    QueueDistributions dists(data_path + "/invariant_distributions_qmax50.csv");
    std::cout << "Loaded queue distributions\n";

    // PowerLawImpact parameters
    // double alpha = 1.5;
    // double A = 2e14; // 3e3
    // double m = 0.4;
    // double eps = 5e10; // 1.51e10

    // LinearImpact parameters
    // double B = 0.4;
    // int64_t t0 = 120LL * 1'000'000'000;  // last metaorder timestamp (120s)

    int num_sims = 100'000;
    int num_threads = std::thread::hardware_concurrency();

    int64_t duration = 20LL * 60 * 1'000'000'000;
    int64_t step = 500'000'000;  // 500ms grid

    std::cout << "Using " << num_threads << " threads\n";
    std::cout << "Grid: " << (duration / step + 1) << " points (500ms spacing)\n";
    std::cout << "NoImpact\n";

    std::filesystem::create_directories(base_results_path);
    Accumulator acc(duration, step);

    std::cout << "Starting (" << num_sims << " simulations)\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (int batch_start = 0; batch_start < num_sims; batch_start += num_threads) {
        int batch_end = std::min(batch_start + num_threads, num_sims);

        std::vector<std::future<void>> futures;
        for (int i = batch_start; i < batch_end; i++) {
            futures.push_back(std::async(std::launch::async, run_and_accumulate,
                                          data_path, std::cref(dists), i, duration, std::ref(acc)));
        }

        for (auto& f : futures) {
            f.get();
        }

        if (batch_start % 10000 == 0) {
            std::cout << "  Progress: " << batch_start << "/" << num_sims << "\n";
        }
    }

    // std::string out_path = base_results_path + "/avg_mid_price_alpha_" +
    //                        std::to_string(alpha) + ".csv";
    // std::string out_path = base_results_path + "/linear_impact_B_" +
    //                        std::to_string(B) + ".csv";
    std::string out_path = base_results_path + "/no_impact.csv";
    acc.save_csv(out_path);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Done in " << elapsed.count() << " seconds\n";
}
