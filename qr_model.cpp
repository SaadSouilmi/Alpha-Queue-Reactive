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

}
