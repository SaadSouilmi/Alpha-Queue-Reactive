#include "qr_model.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace qr {

namespace {
    OrderType parse_event_type(const std::string& s) {
        if (s == "Add") return OrderType::Add;
        if (s == "Can") return OrderType::Cancel;
        if (s == "Trd") return OrderType::Trade;
        if (s == "Create_Bid") return OrderType::CreateBid;
        if (s == "Create_Ask") return OrderType::CreateAsk;
        throw std::runtime_error("Unknown event type: " + s);
    }

    Side parse_side(const std::string& s) {
        if (s == "A") return Side::Ask;
        if (s == "B") return Side::Bid;
        throw std::runtime_error("Unknown side: " + s);
    }

    // Convert imb_bin float value (-1.0, -0.9, ..., 0.0, ..., 0.9, 1.0) to index (0-20)
    int imb_bin_to_index(double imb_bin) {
        // imb_bin: -1.0 -> 0, -0.9 -> 1, ..., 0.0 -> 10, ..., 0.9 -> 19, 1.0 -> 20
        int idx = static_cast<int>(std::round(imb_bin * 10.0)) + 10;
        return std::clamp(idx, 0, 20);
    }
}

QRParams::QRParams(const std::string& path) {
    // Load event probabilities
    // Format: imb_bin,spread,event,event_q,len,event_side,proba
    std::ifstream file(path + "/event_probabilities.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open event_probabilities.csv");
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imb_bin (float like -1.0, -0.9, 0.0, etc.)
        std::getline(ss, token, ',');
        double imb_bin_val = std::stod(token);
        int imb_bin = imb_bin_to_index(imb_bin_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // event type
        std::getline(ss, token, ',');
        OrderType type = parse_event_type(token);

        // event_q (queue number)
        std::getline(ss, token, ',');
        int queue_nbr = std::stoi(token);

        // len (skip - just count)
        std::getline(ss, token, ',');

        // event_side
        std::getline(ss, token, ',');
        Side side = parse_side(token);

        // proba
        std::getline(ss, token, ',');
        double prob = std::stod(token);

        // Add to state params
        StateParams& sp = state_params[imb_bin][spread];
        sp.events.push_back({type, side, queue_nbr});
        sp.base_probs.push_back(prob);
    }
    file.close();

    // Load intensities
    // Format: imb_bin,spread,dt (dt is mean inter-arrival time, so lambda = 1/dt)
    std::ifstream ifile(path + "/intensities.csv");
    if (!ifile.is_open()) {
        throw std::runtime_error("Cannot open intensities.csv");
    }

    std::getline(ifile, line); // skip header

    while (std::getline(ifile, line)) {
        std::istringstream ss(line);
        std::string token;

        // imb_bin (float)
        std::getline(ss, token, ',');
        double imb_bin_val = std::stod(token);
        int imb_bin = imb_bin_to_index(imb_bin_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // dt (mean inter-arrival time)
        std::getline(ss, token, ',');
        double dt_mean = std::stod(token);

        // Store rate (1/mean) for exponential distribution
        state_params[imb_bin][spread].lambda = 1.0 / dt_mean;
    }
    ifile.close();

    // Initialize working vectors for each state
    for (auto& row : state_params) {
        for (auto& sp : row) {
            sp.probs = sp.base_probs;
            sp.cum_probs.resize(sp.probs.size());
            sp.total = 0.0;
            for (size_t i = 0; i < sp.probs.size(); i++) {
                sp.total += sp.probs[i];
            }
            if (!sp.cum_probs.empty()) {
                sp.cum_probs[0] = sp.probs[0];
                for (size_t i = 1; i < sp.probs.size(); i++) {
                    sp.cum_probs[i] = sp.cum_probs[i-1] + sp.probs[i];
                }
            }
        }
    }
}

void QRParams::load_total_lvl_quantiles(const std::string& csv_path) {
    // Load total_lvl quantile edges
    // Format: bin,lower,upper,percentile_lower,percentile_upper
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open total_lvl_quantiles.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    // First edge is the lower of bin 0
    bool first_row = true;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // bin
        std::getline(ss, token, ',');
        int bin = std::stoi(token);

        // lower
        std::getline(ss, token, ',');
        double lower = std::stod(token);

        // upper
        std::getline(ss, token, ',');
        double upper = std::stod(token);

        if (bin >= 0 && bin < NUM_TOTAL_LVL_BINS) {
            if (first_row) {
                total_lvl_edges[0] = lower;
                first_row = false;
            }
            total_lvl_edges[bin + 1] = upper;
        }
    }
    file.close();
}

void QRParams::load_event_probabilities_3d(const std::string& csv_path) {
    // Load 3D event probabilities
    // Format: imb_bin,spread,total_lvl_bin,event,event_q,len,event_side,proba
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open event_probabilities_3d.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imb_bin (float like -1.0, -0.9, 0.0, etc.)
        std::getline(ss, token, ',');
        double imb_bin_val = std::stod(token);
        int imb_bin = imb_bin_to_index(imb_bin_val);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token) - 1; // 1->0, 2->1

        // total_lvl_bin (0-4)
        std::getline(ss, token, ',');
        int total_lvl_bin = std::stoi(token);

        // event type
        std::getline(ss, token, ',');
        OrderType type = parse_event_type(token);

        // event_q (queue number)
        std::getline(ss, token, ',');
        int queue_nbr = std::stoi(token);

        // len (skip - just count)
        std::getline(ss, token, ',');

        // event_side
        std::getline(ss, token, ',');
        Side side = parse_side(token);

        // proba
        std::getline(ss, token, ',');
        double prob = std::stod(token);

        // Validate indices
        if (imb_bin < 0 || imb_bin >= NUM_IMB_BINS) continue;
        if (spread < 0 || spread > 1) continue;
        if (total_lvl_bin < 0 || total_lvl_bin >= NUM_TOTAL_LVL_BINS) continue;

        // Add to 3D state params
        StateParams& sp = event_state_params_3d[imb_bin][spread][total_lvl_bin];
        sp.events.push_back({type, side, queue_nbr});
        sp.base_probs.push_back(prob);
    }
    file.close();

    // Initialize working vectors for each 3D state
    for (auto& imb_row : event_state_params_3d) {
        for (auto& spread_row : imb_row) {
            for (auto& sp : spread_row) {
                sp.probs = sp.base_probs;
                sp.cum_probs.resize(sp.probs.size());
                sp.total = 0.0;
                for (size_t i = 0; i < sp.probs.size(); i++) {
                    sp.total += sp.probs[i];
                }
                if (!sp.cum_probs.empty()) {
                    sp.cum_probs[0] = sp.probs[0];
                    for (size_t i = 1; i < sp.probs.size(); i++) {
                        sp.cum_probs[i] = sp.cum_probs[i-1] + sp.probs[i];
                    }
                }
            }
        }
    }

    use_total_lvl = true;
}

SizeDistributions::SizeDistributions(const std::string& csv_path) {
    // Load size distribution parameters
    // Format: imb_bin,event,event_q,p,spread
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open size_distrib.csv: " + csv_path);
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // imb_bin (float like -1.0, -0.9, 0.0, etc.)
        std::getline(ss, token, ',');
        double imb_bin_val = std::stod(token);
        int imb_bin = imb_bin_to_index(imb_bin_val);

        // event type
        std::getline(ss, token, ',');
        std::string event_str = token;

        // event_q (queue number: 0, 1, or 2)
        std::getline(ss, token, ',');
        int queue = std::stoi(token);

        // p (geometric distribution parameter)
        std::getline(ss, token, ',');
        double p = std::stod(token);

        // spread (1 or 2)
        std::getline(ss, token, ',');
        int spread = std::stoi(token);

        if (imb_bin < 0 || imb_bin >= NUM_IMB_BINS) continue;

        if (spread == 1) {
            // Spread=1: Add, Can, Trd at queue 1 or 2
            int type_idx;
            if (event_str == "Add") type_idx = 0;
            else if (event_str == "Can") type_idx = 1;
            else if (event_str == "Trd") type_idx = 2;
            else continue;

            int queue_idx = queue - 1;  // 1->0, 2->1
            if (queue_idx >= 0 && queue_idx < 2) {
                p_params[imb_bin][type_idx][queue_idx] = p;
            }
        } else if (spread == 2) {
            // Spread>=2: Create_Bid, Create_Ask
            int create_idx;
            if (event_str == "Create_Bid") create_idx = 0;
            else if (event_str == "Create_Ask") create_idx = 1;
            else continue;

            p_create[imb_bin][create_idx] = p;
        }
    }
    file.close();
}

}
