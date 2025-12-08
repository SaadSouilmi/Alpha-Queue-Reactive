#include <iostream>
#include <chrono>
#include "orderbook.h"
#include "qr_model.h"
#include "simulation.h"

using namespace qr;

int main() {
    std::string data_path = "/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2";
    std::string output_path = "/Users/saad.souilmi/dev_cpp/queue_reactive/result.parquet";

    // Load queue distributions
    QueueDistributions dists(data_path + "/inv_distributions_qmax30.csv");

    // Initialize order book
    OrderBook lob(dists, 4, 73892742);
    lob.init({1516, 1517, 1518, 1519},
              {4, 1, 10, 5},
              {1520, 1521, 1522, 1523},
              {6, 17, 22, 23});

    QRParams params(data_path);
    QRModel model(&lob, params, 42);

    int64_t duration = 1e9 * 3600 * 1000;  // 5.5 hours

    auto start = std::chrono::high_resolution_clock::now();
    Buffer result = run_simple(lob, model, duration);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simulation: " << elapsed.count() << " ms\n";
    std::cout << "Events: " << result.num_events() << "\n";

    auto start_save = std::chrono::high_resolution_clock::now();
    result.save_parquet(output_path);
    auto end_save = std::chrono::high_resolution_clock::now();

    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);
    std::cout << "Parquet save: " << save_time.count() << " ms\n";
}
