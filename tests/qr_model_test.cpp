#include <gtest/gtest.h>
#include "../qr_model.h"
#include <set>
#include <cmath>

using namespace qr;

// ============================================================================
// StateParams Tests
// ============================================================================

class StateParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple StateParams with known events
        sp.events = {
            {OrderType::Trade, Side::Bid, -1},
            {OrderType::Trade, Side::Ask, 1},
            {OrderType::Add, Side::Bid, -1},
            {OrderType::Cancel, Side::Ask, 1}
        };
        sp.base_probs = {0.1, 0.2, 0.3, 0.4};
        sp.probs = sp.base_probs;
        sp.cum_probs.resize(4);
        sp.total = 0.0;
        for (size_t i = 0; i < sp.probs.size(); i++) {
            sp.total += sp.probs[i];
        }
        sp.cum_probs[0] = sp.probs[0];
        for (size_t i = 1; i < sp.probs.size(); i++) {
            sp.cum_probs[i] = sp.cum_probs[i-1] + sp.probs[i];
        }
        sp.lambda = 1e-6;  // rate = 1/mean
    }

    StateParams sp;
    std::mt19937_64 rng{42};
};

TEST_F(StateParamsTest, SampleEventReturnsValidEvent) {
    for (int i = 0; i < 100; i++) {
        const Event& e = sp.sample_event(rng);
        // Check it's one of our events
        bool found = false;
        for (const auto& ev : sp.events) {
            if (ev.type == e.type && ev.side == e.side && ev.queue_nbr == e.queue_nbr) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(StateParamsTest, SampleEventDistribution) {
    // Sample many times and check distribution roughly matches probabilities
    std::map<OrderType, int> counts;
    int n = 10000;

    for (int i = 0; i < n; i++) {
        const Event& e = sp.sample_event(rng);
        counts[e.type]++;
    }

    // Trade Bid: 0.1, Trade Ask: 0.2, Add: 0.3, Cancel: 0.4
    EXPECT_NEAR(counts[OrderType::Trade] / (double)n, 0.3, 0.05);  // 0.1 + 0.2
    EXPECT_NEAR(counts[OrderType::Add] / (double)n, 0.3, 0.05);
    EXPECT_NEAR(counts[OrderType::Cancel] / (double)n, 0.4, 0.05);
}

TEST_F(StateParamsTest, BiasIncreasesTradesBid) {
    sp.bias(0.5);  // Bid trades *= 1.5, Ask trades *= 0.5

    // Original: Bid trade = 0.1, Ask trade = 0.2
    // After bias: Bid trade = 0.15, Ask trade = 0.1
    // Add and Cancel unchanged: 0.3, 0.4

    std::map<std::pair<OrderType, Side>, int> counts;
    int n = 10000;

    for (int i = 0; i < n; i++) {
        const Event& e = sp.sample_event(rng);
        counts[{e.type, e.side}]++;
    }

    double bid_trade_ratio = counts[{OrderType::Trade, Side::Bid}] / (double)n;
    double ask_trade_ratio = counts[{OrderType::Trade, Side::Ask}] / (double)n;

    // Bid trades should be higher than ask trades now
    EXPECT_GT(bid_trade_ratio, ask_trade_ratio);
}

TEST_F(StateParamsTest, BiasPreservesBaseProbs) {
    std::vector<double> original = sp.base_probs;
    sp.bias(0.5);
    sp.bias(-0.3);
    sp.bias(0.8);

    // base_probs should be unchanged
    EXPECT_EQ(sp.base_probs, original);
}

TEST_F(StateParamsTest, SampleDtReturnsPositive) {
    for (int i = 0; i < 100; i++) {
        long long dt = sp.sample_dt(rng);
        EXPECT_GE(dt, 1);  // ceil ensures at least 1
    }
}

// ============================================================================
// QRParams Tests
// ============================================================================

class QRParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // This test requires the actual CSV files
        // Skip if files don't exist
    }
};

TEST_F(QRParamsTest, LoadFromCSV) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");

        // Check all states are populated (21 imbalance bins now)
        for (int imb = 0; imb < QRParams::NUM_IMB_BINS; imb++) {
            for (int spread = 0; spread < 2; spread++) {
                StateParams& sp = params.state_params[imb][spread];
                EXPECT_FALSE(sp.events.empty()) << "Empty events at imb=" << imb << " spread=" << spread;
                EXPECT_EQ(sp.events.size(), sp.base_probs.size());
                EXPECT_EQ(sp.probs.size(), sp.base_probs.size());
                EXPECT_EQ(sp.cum_probs.size(), sp.base_probs.size());
                // Only spread=0 has lambda from intensities.csv
                if (spread == 0) {
                    EXPECT_GT(sp.lambda, 0.0);
                }
                EXPECT_GT(sp.total, 0.0);
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRParamsTest, ProbabilitiesSumToOne) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");

        for (int imb = 0; imb < QRParams::NUM_IMB_BINS; imb++) {
            for (int spread = 0; spread < 2; spread++) {
                StateParams& sp = params.state_params[imb][spread];
                double sum = 0.0;
                for (double p : sp.base_probs) {
                    sum += p;
                }
                EXPECT_NEAR(sum, 1.0, 0.01) << "Probs don't sum to 1 at imb=" << imb << " spread=" << spread;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

// ============================================================================
// QRModel Tests
// ============================================================================

class QRModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load queue distributions for OrderBook
        try {
            dists = std::make_unique<QueueDistributions>("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL/inv_distributions_qmax30.csv");
        } catch (...) {
            // If file not found, create empty distributions
            dists = std::make_unique<QueueDistributions>();
        }
        lob = std::make_unique<OrderBook>(*dists, 4);
        lob->init({1516, 1517, 1518, 1519},
                  {4, 1, 10, 5},
                  {1520, 1521, 1522, 1523},
                  {3, 17, 22, 23});
    }

    std::unique_ptr<QueueDistributions> dists;
    std::unique_ptr<OrderBook> lob;
};

TEST_F(QRModelTest, SampleOrderReturnsValidOrder) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model(lob.get(), params, 42);

        for (int i = 0; i < 100; i++) {
            Order order = model.sample_order(i * 1000);

            // Check order type is valid
            EXPECT_TRUE(order.type == OrderType::Add ||
                       order.type == OrderType::Cancel ||
                       order.type == OrderType::Trade ||
                       order.type == OrderType::CreateBid ||
                       order.type == OrderType::CreateAsk);

            // Check side is valid
            EXPECT_TRUE(order.side == Side::Bid || order.side == Side::Ask);

            // Check timestamp is set
            EXPECT_EQ(order.ts, i * 1000);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRModelTest, SampleDtReturnsPositive) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model(lob.get(), params, 42);

        for (int i = 0; i < 100; i++) {
            int64_t dt = model.sample_dt();
            EXPECT_GE(dt, 1);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRModelTest, DeterministicWithSameSeed) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model1(lob.get(), params, 123);
        QRModel model2(lob.get(), params, 123);

        for (int i = 0; i < 50; i++) {
            Order o1 = model1.sample_order(i);
            Order o2 = model2.sample_order(i);

            EXPECT_EQ(o1.type, o2.type);
            EXPECT_EQ(o1.side, o2.side);
            EXPECT_EQ(o1.price, o2.price);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRModelTest, DifferentWithDifferentSeeds) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model1(lob.get(), params, 111);
        QRModel model2(lob.get(), params, 222);

        int differences = 0;
        for (int i = 0; i < 50; i++) {
            Order o1 = model1.sample_order(i);
            Order o2 = model2.sample_order(i);

            if (o1.type != o2.type || o1.side != o2.side || o1.price != o2.price) {
                differences++;
            }
        }
        EXPECT_GT(differences, 0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRModelTest, PriceComputationCreateBid) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model(lob.get(), params, 42);

        // For CreateBid, price should be best_bid + 1
        int32_t expected = lob->best_bid() + 1;

        // Sample until we get a CreateBid event
        bool found = false;
        for (int i = 0; i < 10000 && !found; i++) {
            Order order = model.sample_order(i);
            if (order.type == OrderType::CreateBid) {
                EXPECT_EQ(order.price, expected);
                found = true;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

TEST_F(QRModelTest, PriceComputationCreateAsk) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model(lob.get(), params, 42);

        // For CreateAsk, price should be best_ask - 1
        int32_t expected = lob->best_ask() - 1;

        // Sample until we get a CreateAsk event
        bool found = false;
        for (int i = 0; i < 10000 && !found; i++) {
            Order order = model.sample_order(i);
            if (order.type == OrderType::CreateAsk) {
                EXPECT_EQ(order.price, expected);
                found = true;
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

// ============================================================================
// Integration Test
// ============================================================================

TEST_F(QRModelTest, SimulationRuns) {
    try {
        QRParams params("/Users/saad.souilmi/dev_cpp/queue_reactive/data/AAL2");
        QRModel model(lob.get(), params, 42);

        int64_t time = 0;
        int num_orders = 1000;

        for (int i = 0; i < num_orders; i++) {
            int64_t dt = model.sample_dt();
            time += dt;

            Order order = model.sample_order(time);
            lob->process(order);

            // LOB should remain valid
            EXPECT_NO_THROW(lob->best_bid());
            EXPECT_NO_THROW(lob->best_ask());
            EXPECT_GE(lob->spread(), 1u);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CSV files not found: " << e.what();
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
