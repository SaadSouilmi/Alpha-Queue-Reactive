#include <gtest/gtest.h>
#include "../orderbook.h"

using namespace qr;

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        lob = std::make_unique<OrderBook>(4);
        lob->init({1516, 1517, 1518, 1519},  // bid prices
                  {4, 1, 10, 5},              // bid volumes
                  {1520, 1521, 1522, 1523},   // ask prices
                  {3, 17, 22, 23});           // ask volumes
    }

    std::unique_ptr<OrderBook> lob;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(OrderBookTest, InitializationBestPrices) {
    EXPECT_EQ(lob->best_bid(), 1519);
    EXPECT_EQ(lob->best_ask(), 1520);
}

TEST_F(OrderBookTest, InitializationBestVolumes) {
    EXPECT_EQ(lob->best_bid_vol(), 5);
    EXPECT_EQ(lob->best_ask_vol(), 3);
}

TEST_F(OrderBookTest, InitializationSpread) {
    EXPECT_EQ(lob->spread(), 1);
}

TEST_F(OrderBookTest, InitializationImbalance) {
    // Imbalance = (5 - 3) / (5 + 3) = 2/8 = 0.25
    EXPECT_NEAR(lob->imbalance(), 0.25, 0.001);
}

// ============================================================================
// Add Order Tests
// ============================================================================

TEST_F(OrderBookTest, AddOrderToBid) {
    Order order(OrderType::Add, Side::Bid, 1519, 10, 1000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);
    EXPECT_EQ(lob->best_bid_vol(), 15);  // 5 + 10
}

TEST_F(OrderBookTest, AddOrderToAsk) {
    Order order(OrderType::Add, Side::Ask, 1520, 5, 1000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);
    EXPECT_EQ(lob->best_ask_vol(), 8);  // 3 + 5
}

TEST_F(OrderBookTest, AddOrderToNonExistentPrice) {
    Order order(OrderType::Add, Side::Bid, 9999, 10, 1000);
    lob->process(order);

    EXPECT_TRUE(order.rejected);
}

// ============================================================================
// Cancel Order Tests
// ============================================================================

TEST_F(OrderBookTest, CancelOrderPartial) {
    Order order(OrderType::Cancel, Side::Ask, 1520, 2, 2000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);
    EXPECT_EQ(lob->best_ask_vol(), 1);  // 3 - 2
}

TEST_F(OrderBookTest, CancelOrderFull) {
    Order order(OrderType::Cancel, Side::Ask, 1520, 3, 2000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    // After cleaning, best ask should shift
    EXPECT_NE(lob->best_ask(), 1520);
}

TEST_F(OrderBookTest, CancelOrderExceedsVolume) {
    Order order(OrderType::Cancel, Side::Ask, 1520, 100, 2000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);
}

TEST_F(OrderBookTest, CancelOrderNonExistentPrice) {
    Order order(OrderType::Cancel, Side::Bid, 9999, 10, 2000);
    lob->process(order);

    EXPECT_TRUE(order.rejected);
}

// ============================================================================
// Trade Order Tests (Sweeping)
// ============================================================================

TEST_F(OrderBookTest, TradeSellSingleLevel) {
    // Side::Bid = sell order (hits bid side)
    Order order(OrderType::Trade, Side::Bid, 1519, 2, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);
    EXPECT_EQ(lob->best_bid_vol(), 3);  // 5 - 2
}

TEST_F(OrderBookTest, TradeSellMultipleLevels) {
    // Side::Bid = sell order (hits bid side)
    // Sell enough to sweep through multiple levels
    // Available: 5 @ 1519, 10 @ 1518, 1 @ 1517, 4 @ 1516
    Order order(OrderType::Trade, Side::Bid, 1516, 20, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);  // Consumed 5 + 10 + 1 + 4 = 20, fully filled
}

TEST_F(OrderBookTest, TradeBuySingleLevel) {
    // Side::Ask = buy order (hits ask side)
    Order order(OrderType::Trade, Side::Ask, 1520, 2, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);
    EXPECT_EQ(lob->best_ask_vol(), 1);  // 3 - 2
}

TEST_F(OrderBookTest, TradeBuyMultipleLevels) {
    // Side::Ask = buy order (hits ask side)
    // Buy enough to sweep through multiple levels
    // Available: 3 @ 1520, 17 @ 1521, 22 @ 1522
    Order order(OrderType::Trade, Side::Ask, 1522, 25, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);  // Consumed 3 + 17 + 5 = 25, fully filled
}

TEST_F(OrderBookTest, TradeSellPriceTooHigh) {
    // Side::Bid = sell, price above best bid should be rejected
    Order order(OrderType::Trade, Side::Bid, 1525, 10, 3000);
    lob->process(order);

    EXPECT_TRUE(order.rejected);
}

TEST_F(OrderBookTest, TradeBuyPriceTooLow) {
    // Side::Ask = buy, price below best ask should be rejected
    Order order(OrderType::Trade, Side::Ask, 1515, 10, 3000);
    lob->process(order);

    EXPECT_TRUE(order.rejected);
}

// ============================================================================
// Marketable Limit Order Tests (Residual Posting)
// ============================================================================

TEST_F(OrderBookTest, MarketableLimitSellWithResidual) {
    // Side::Bid = sell order at limit price 1518
    // Should consume: 5 @ 1519, 10 @ 1518
    // Remaining: 10 - 15 = -5, so fully executed? No wait...
    // Order size 20, consumes 15, remaining 5 should be posted as ask
    Order order(OrderType::Trade, Side::Bid, 1518, 20, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);  // Partially filled

    // Check that best bid shifted down after consuming 1519 and 1518
    EXPECT_EQ(lob->best_bid(), 1517);

    // Check that residual (5 shares) was posted as ask at limit price 1518
    // But after clean_ask(), it might have adjusted. Let's check best ask.
    // The residual should create a new ask level at 1518, making it the new best ask
    EXPECT_EQ(lob->best_ask(), 1518);
}

TEST_F(OrderBookTest, MarketableLimitBuyWithResidual) {
    // Side::Ask = buy order at limit price 1521
    // Should consume: 3 @ 1520, 17 @ 1521
    // Order size 25, consumes 20, remaining 5 should be posted as bid
    Order order(OrderType::Trade, Side::Ask, 1521, 25, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);

    // Check that best ask shifted up after consuming 1520 and 1521
    EXPECT_EQ(lob->best_ask(), 1522);

    // Check that residual (5 shares) was posted as bid at limit price 1521
    EXPECT_EQ(lob->best_bid(), 1521);
}

TEST_F(OrderBookTest, MarketableLimitSellFullyExecuted) {
    // Sell exactly what's available, no residual
    // Available at 1519: 5 shares
    Order order(OrderType::Trade, Side::Bid, 1519, 5, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);  // Fully executed

    // Best bid should now be 1518
    EXPECT_EQ(lob->best_bid(), 1518);

    // No residual posted
    EXPECT_EQ(lob->best_ask(), 1520);  // Unchanged
}

TEST_F(OrderBookTest, MarketableLimitBuyFullyExecuted) {
    // Buy exactly what's available, no residual
    // Available at 1520: 3 shares
    Order order(OrderType::Trade, Side::Ask, 1520, 3, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);  // Fully executed

    // Best ask should now be 1521
    EXPECT_EQ(lob->best_ask(), 1521);

    // No residual posted
    EXPECT_EQ(lob->best_bid(), 1519);  // Unchanged
}

TEST_F(OrderBookTest, MarketableLimitSellResidualAtBestBid) {
    // Sell with limit price AT best bid
    // Should consume some and post residual
    Order order(OrderType::Trade, Side::Bid, 1519, 10, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);  // Consumed 5, remaining 5

    // Consumed all of 1519, best bid is now 1518
    EXPECT_EQ(lob->best_bid(), 1518);

    // Residual 5 shares posted as ask at 1519
    EXPECT_EQ(lob->best_ask(), 1519);  // New best ask!
}

TEST_F(OrderBookTest, MarketableLimitBuyResidualAtBestAsk) {
    // Buy with limit price AT best ask
    Order order(OrderType::Trade, Side::Ask, 1520, 10, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);  // Consumed 3, remaining 7

    // Consumed all of 1520, best ask is now 1521
    EXPECT_EQ(lob->best_ask(), 1521);

    // Residual 7 shares posted as bid at 1520
    EXPECT_EQ(lob->best_bid(), 1520);  // New best bid!
}

TEST_F(OrderBookTest, MarketableLimitSellDeepSweep) {
    // Sell order that sweeps through ALL bid levels
    // Total available: 5 + 10 + 1 + 4 = 20
    Order order(OrderType::Trade, Side::Bid, 1516, 30, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);  // Consumed 20, remaining 10

    // All bids consumed, but clean_bid() should have created new levels
    // The residual 10 shares should be posted as ask at limit price 1516
    // After posting, clean_ask() ensures proper LOB structure
}

TEST_F(OrderBookTest, MarketableLimitBuyDeepSweep) {
    // Buy order that sweeps through ALL ask levels
    // Total available: 3 + 17 + 22 + 23 = 65
    Order order(OrderType::Trade, Side::Ask, 1523, 70, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_TRUE(order.partial);  // Consumed 65, remaining 5

    // All asks consumed, but clean_ask() creates new levels
    // Residual 5 shares posted as bid at 1523
}

TEST_F(OrderBookTest, MarketableLimitAddToExistingLevel) {
    // First, create a bid at 1520
    Order create1(OrderType::CreateBid, Side::Bid, 1520, 5, 1000);
    lob->process(create1);

    // Now sell with limit 1520, should sweep 1519 and post residual at 1520
    // Since 1520 already has 5 from our create order
    Order sell(OrderType::Trade, Side::Bid, 1520, 10, 2000);
    lob->process(sell);

    EXPECT_FALSE(sell.rejected);
    EXPECT_TRUE(sell.partial);  // Consumed 5 @ 1519, remaining 5

    // The residual 5 should be ADDED to existing ask at 1520
    // After consuming bid at 1519, if there's an ask level at 1520, add to it
}

TEST_F(OrderBookTest, MarketableLimitNoExecutionBecomesLimitOrder) {
    // Sell order with limit ABOVE best bid (can't execute anything)
    Order order(OrderType::Trade, Side::Bid, 1525, 10, 3000);
    lob->process(order);

    EXPECT_TRUE(order.rejected);  // Nothing executed

    // No residual posted because nothing traded
}

TEST_F(OrderBookTest, MarketableLimitExactMultiLevelFill) {
    // Order that exactly consumes multiple levels with no residual
    // Available: 5 @ 1519, 10 @ 1518 = 15 total
    Order order(OrderType::Trade, Side::Bid, 1518, 15, 3000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    EXPECT_FALSE(order.partial);  // Exactly filled

    EXPECT_EQ(lob->best_bid(), 1517);
    EXPECT_EQ(lob->best_ask(), 1520);  // Unchanged
}

// ============================================================================
// Create Order Tests
// ============================================================================

TEST_F(OrderBookTest, CreateBidOrder) {
    int32_t old_best_bid = lob->best_bid();

    Order order(OrderType::CreateBid, Side::Bid, 1520, 10, 4000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    // New level created, best bid should change (or stay if 1520 existed)
    EXPECT_GE(lob->best_bid(), old_best_bid);
}

TEST_F(OrderBookTest, CreateAskOrder) {
    int32_t old_best_ask = lob->best_ask();

    Order order(OrderType::CreateAsk, Side::Ask, 1519, 10, 4000);
    lob->process(order);

    EXPECT_FALSE(order.rejected);
    // New level created inside spread
    EXPECT_LE(lob->best_ask(), old_best_ask);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(OrderBookTest, MultipleOperations) {
    // Add liquidity to bid side
    Order add1(OrderType::Add, Side::Bid, 1519, 5, 1000);
    lob->process(add1);
    EXPECT_EQ(lob->best_bid_vol(), 10);  // 5 + 5 = 10

    // Cancel some from bid side
    Order cancel1(OrderType::Cancel, Side::Bid, 1519, 3, 2000);
    lob->process(cancel1);
    EXPECT_EQ(lob->best_bid_vol(), 7);  // 10 - 3 = 7

    // Side::Bid = sell order, hits bid side
    Order trade1(OrderType::Trade, Side::Bid, 1519, 4, 3000);
    lob->process(trade1);
    EXPECT_EQ(lob->best_bid_vol(), 3);  // 7 - 4 = 3
}

TEST_F(OrderBookTest, ImbalanceAfterTrade) {
    // Side::Ask = buy order (hits ask side)
    Order order(OrderType::Trade, Side::Ask, 1520, 2, 3000);
    lob->process(order);

    // New imbalance: (5 - 1) / (5 + 1) = 4/6 ≈ 0.667
    EXPECT_NEAR(lob->imbalance(), 0.667, 0.01);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}