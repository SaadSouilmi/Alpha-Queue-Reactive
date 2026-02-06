#include "simulation.h"
#include "race.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <fstream>

namespace qr {

void Buffer::save_parquet(const std::string& path) const {
    // Build columns
    arrow::Int64Builder sequence_builder;
    arrow::Int32Builder best_bid_price_builder;
    arrow::Int32Builder best_bid_vol_builder;
    arrow::Int32Builder best_ask_price_builder;
    arrow::Int32Builder best_ask_vol_builder;
    arrow::Int32Builder second_bid_price_builder;
    arrow::Int32Builder second_bid_vol_builder;
    arrow::Int32Builder second_ask_price_builder;
    arrow::Int32Builder second_ask_vol_builder;
    arrow::DoubleBuilder imbalance_builder;
    arrow::DoubleBuilder mid_builder;
    arrow::Int64Builder timestamp_builder;
    arrow::StringBuilder type_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder price_builder;
    arrow::Int32Builder volume_builder;
    arrow::BooleanBuilder rejected_builder;
    arrow::BooleanBuilder partial_builder;
    arrow::DoubleBuilder bias_builder;
    arrow::DoubleBuilder alpha_builder;
    arrow::Int8Builder source_builder;

    for (const auto& r : records) {
        (void)sequence_builder.Append(r.sequence);
        (void)best_bid_price_builder.Append(r.best_bid_price);
        (void)best_bid_vol_builder.Append(r.best_bid_vol);
        (void)best_ask_price_builder.Append(r.best_ask_price);
        (void)best_ask_vol_builder.Append(r.best_ask_vol);
        (void)second_bid_price_builder.Append(r.second_bid_price);
        (void)second_bid_vol_builder.Append(r.second_bid_vol);
        (void)second_ask_price_builder.Append(r.second_ask_price);
        (void)second_ask_vol_builder.Append(r.second_ask_vol);
        (void)imbalance_builder.Append(r.imbalance);
        (void)mid_builder.Append(r.mid);
        (void)timestamp_builder.Append(r.timestamp);
        (void)type_builder.Append(r.type);
        (void)side_builder.Append(static_cast<int8_t>(r.side));
        (void)price_builder.Append(r.price);
        (void)volume_builder.Append(r.volume);
        (void)rejected_builder.Append(r.rejected);
        (void)partial_builder.Append(r.partial);
        (void)bias_builder.Append(r.bias);
        (void)alpha_builder.Append(r.alpha);
        (void)source_builder.Append(r.source);
    }

    std::shared_ptr<arrow::Array> sequence_arr;
    std::shared_ptr<arrow::Array> best_bid_price_arr, best_bid_vol_arr;
    std::shared_ptr<arrow::Array> best_ask_price_arr, best_ask_vol_arr;
    std::shared_ptr<arrow::Array> second_bid_price_arr, second_bid_vol_arr;
    std::shared_ptr<arrow::Array> second_ask_price_arr, second_ask_vol_arr;
    std::shared_ptr<arrow::Array> imbalance_arr;
    std::shared_ptr<arrow::Array> mid_arr;
    std::shared_ptr<arrow::Array> timestamp_arr, type_arr, side_arr;
    std::shared_ptr<arrow::Array> price_arr, volume_arr, rejected_arr, partial_arr;
    std::shared_ptr<arrow::Array> bias_arr, alpha_arr;
    std::shared_ptr<arrow::Array> source_arr;

    (void)sequence_builder.Finish(&sequence_arr);
    (void)best_bid_price_builder.Finish(&best_bid_price_arr);
    (void)best_bid_vol_builder.Finish(&best_bid_vol_arr);
    (void)best_ask_price_builder.Finish(&best_ask_price_arr);
    (void)best_ask_vol_builder.Finish(&best_ask_vol_arr);
    (void)second_bid_price_builder.Finish(&second_bid_price_arr);
    (void)second_bid_vol_builder.Finish(&second_bid_vol_arr);
    (void)second_ask_price_builder.Finish(&second_ask_price_arr);
    (void)second_ask_vol_builder.Finish(&second_ask_vol_arr);
    (void)imbalance_builder.Finish(&imbalance_arr);
    (void)mid_builder.Finish(&mid_arr);
    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)type_builder.Finish(&type_arr);
    (void)side_builder.Finish(&side_arr);
    (void)price_builder.Finish(&price_arr);
    (void)volume_builder.Finish(&volume_arr);
    (void)rejected_builder.Finish(&rejected_arr);
    (void)partial_builder.Finish(&partial_arr);
    (void)bias_builder.Finish(&bias_arr);
    (void)alpha_builder.Finish(&alpha_arr);
    (void)source_builder.Finish(&source_arr);

    auto schema = arrow::schema({
        arrow::field("sequence", arrow::int64()),
        arrow::field("p_-1", arrow::int32()),
        arrow::field("q_-1", arrow::int32()),
        arrow::field("p_1", arrow::int32()),
        arrow::field("q_1", arrow::int32()),
        arrow::field("p_-2", arrow::int32()),
        arrow::field("q_-2", arrow::int32()),
        arrow::field("p_2", arrow::int32()),
        arrow::field("q_2", arrow::int32()),
        arrow::field("imbalance", arrow::float64()),
        arrow::field("mid", arrow::float64()),
        arrow::field("ts_event", arrow::int64()),
        arrow::field("event", arrow::utf8()),
        arrow::field("side", arrow::int8()),
        arrow::field("price", arrow::int32()),
        arrow::field("size", arrow::int32()),
        arrow::field("rejected", arrow::boolean()),
        arrow::field("partial", arrow::boolean()),
        arrow::field("bias", arrow::float64()),
        arrow::field("alpha", arrow::float64()),
        arrow::field("source", arrow::int8())
    });

    auto table = arrow::Table::Make(schema, {
        sequence_arr,
        best_bid_price_arr, best_bid_vol_arr,
        best_ask_price_arr, best_ask_vol_arr,
        second_bid_price_arr, second_bid_vol_arr,
        second_ask_price_arr, second_ask_vol_arr,
        imbalance_arr,
        mid_arr,
        timestamp_arr, type_arr, side_arr,
        price_arr, volume_arr, rejected_arr, partial_arr,
        bias_arr, alpha_arr, source_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, records.size());
}

void StrategyBuffer::save_parquet(const std::string& path) const {
    arrow::Int64Builder timestamp_builder;
    arrow::Int32Builder inventory_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder size_builder;
    arrow::Int32Builder price_builder;
    arrow::BooleanBuilder rejected_builder;
    arrow::DoubleBuilder pnl_builder;

    for (const auto& t : trades) {
        (void)timestamp_builder.Append(t.timestamp);
        (void)inventory_builder.Append(t.inventory);
        (void)side_builder.Append(static_cast<int8_t>(t.side));
        (void)size_builder.Append(t.size);
        (void)price_builder.Append(t.price);
        (void)rejected_builder.Append(t.rejected);
        (void)pnl_builder.Append(t.pnl);
    }

    std::shared_ptr<arrow::Array> timestamp_arr, inventory_arr, side_arr;
    std::shared_ptr<arrow::Array> size_arr, price_arr, rejected_arr, pnl_arr;

    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)inventory_builder.Finish(&inventory_arr);
    (void)side_builder.Finish(&side_arr);
    (void)size_builder.Finish(&size_arr);
    (void)price_builder.Finish(&price_arr);
    (void)rejected_builder.Finish(&rejected_arr);
    (void)pnl_builder.Finish(&pnl_arr);

    auto schema = arrow::schema({
        arrow::field("timestamp", arrow::int64()),
        arrow::field("inventory", arrow::int32()),
        arrow::field("side", arrow::int8()),
        arrow::field("size", arrow::int32()),
        arrow::field("price", arrow::int32()),
        arrow::field("rejected", arrow::boolean()),
        arrow::field("pnl", arrow::float64())
    });

    auto table = arrow::Table::Make(schema, {
        timestamp_arr, inventory_arr, side_arr, size_arr, price_arr, rejected_arr, pnl_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, trades.size());
}

Buffer run_simple(OrderBook& lob, QRModel& model, int64_t duration) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    while (time < duration) {
        int64_t dt = model.sample_dt();
        time += dt;

        if (time >= duration) break;

        model.bias(0.0);
        Order order = model.sample_order(time);

        // Capture LOB state before processing
        EventRecord base_record;
        base_record.record_lob(lob);

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;

            // Create a record for each fill (same sequence, same timestamp)
            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Trade";
                record.side = order.side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_QR;

                filled_size += fill.size;
                buffer.records.push_back(record);
            }

            // If partial, add a record for the resting limit order (same sequence)
            // Resting order is on the opposite side (buy rests as bid, sell rests as ask)
            if (order.partial) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Add";
                record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                record.price = order.price;
                record.volume = order.size - filled_size;
                record.rejected = false;
                record.partial = true;
                record.source = SOURCE_QR;

                buffer.records.push_back(record);
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            record.source = SOURCE_QR;

            buffer.records.push_back(record);
        }
    }

    return buffer;
}

Buffer run_metaorder(OrderBook& lob, QRModel& model, MarketImpact& impact, MetaOrder& metaorder, int64_t duration) {
    Buffer buffer;
    int64_t time = 0;
    size_t i = 0;
    size_t n = metaorder.timestamps.size();
    double current_bias = 0.0;

    Order order;
    while (time < duration) {
        impact.step(time);
        current_bias = impact.bias_factor();
        model.bias(current_bias);

        int64_t dt = model.sample_dt();
        if (i < n && time + dt >= metaorder.timestamps[i]) {
            time = metaorder.timestamps[i];
            int32_t price = (metaorder.side == Side::Bid) ? lob.best_bid() : lob.best_ask();
            order = Order(OrderType::Trade, metaorder.side, price, metaorder.sizes[i], time);
            i++;
        }
        else {
            time += dt;
            order = model.sample_order(time);
        }

        EventRecord record;
        record.record_lob(lob);
        lob.process(order);
        record.record_order(order);
        record.bias = current_bias;

        if (order.type == OrderType::Trade) {
						impact.step(order.ts);
						impact.add_trade(order.side, order.size);
            //impact.update(order.side, order.ts, order.size);
        }
        record.source = SOURCE_QR;

        buffer.records.push_back(record);
    }
    return buffer;
}

void AlphaPnL::save_csv(const std::string& path) const {
    std::ofstream file(path);
    file << "lag_sec,alpha_tickreturn_cov,alpha_tickreturn_cov_ci\n";
    for (size_t i = 0; i < lag_sec.size(); i++) {
        file << lag_sec[i] << "," << alpha_tickreturn_cov[i] << "," << alpha_tickreturn_cov_ci[i] << "\n";
    }
}

AlphaPnL compute_alpha_pnl(const Buffer& buffer, const std::vector<int64_t>& lags_ns) {
    AlphaPnL result;
    const auto& records = buffer.records;
    size_t n = records.size();

    // Extract arrays
    std::vector<int64_t> timestamps(n);
    std::vector<double> mid(n);
    std::vector<double> alpha(n);

    for (size_t i = 0; i < n; i++) {
        timestamps[i] = records[i].timestamp;
        mid[i] = records[i].mid;
        alpha[i] = records[i].alpha;
    }

    // Compute for each lag
    for (int64_t lag : lags_ns) {
        // Welford's online algorithm for mean and variance
        double mean = 0.0;
        double M2 = 0.0;
        size_t count = 0;
        size_t j = 0;

        for (size_t i = 0; i < n; i++) {
            if (records[i].rejected) continue;
            int64_t target = timestamps[i] + lag;
            while (j < n && timestamps[j] < target) j++;
            while (j < n && records[j].rejected) j++;
            if (j < n && mid[i] > 0.0 && mid[j] > 0.0) {
                double tick_ret = mid[j] - mid[i];
                double sample = alpha[i] * tick_ret;
                count++;
                double delta = sample - mean;
                mean += delta / static_cast<double>(count);
                double delta2 = sample - mean;
                M2 += delta * delta2;
            }
        }

        result.lag_sec.push_back(static_cast<double>(lag) / 1e9);
        result.alpha_tickreturn_cov.push_back(mean);

        // Compute 95% CI half-width: 1.96 * SE = 1.96 * sqrt(var / n)
        double ci = 0.0;
        if (count > 1) {
            double variance = M2 / static_cast<double>(count - 1);
            double std_err = std::sqrt(variance / static_cast<double>(count));
            ci = 1.96 * std_err;
        }
        result.alpha_tickreturn_cov_ci.push_back(ci);
    }

    return result;
}

Buffer run_with_alpha(OrderBook& lob, QRModel& model, MarketImpact& impact, Alpha& alpha, int64_t duration, double alpha_scale) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    while (time < duration) {
        // Sample dt and advance alpha process
        int64_t dt = model.sample_dt();
        alpha.step(dt);
        time += dt;

        if (time >= duration) break;

        // Combine alpha + impact bias
        // Negative alpha because positive alpha should push price UP
        // (positive bias_factor increases bid trades which pushes price DOWN)
        double alpha_val = alpha.value();
        impact.step(time);
        double impact_val = impact.bias_factor();
        double total_bias = -alpha_val * alpha_scale + impact_val;
        model.bias(total_bias);

        Order order = model.sample_order(time);

        // Capture LOB state before processing
        EventRecord base_record;
        base_record.record_lob(lob);
        base_record.bias = total_bias;
        base_record.alpha = alpha_val;

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;

            // Create a record for each fill (same sequence, same timestamp)
            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Trade";
                record.side = order.side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_QR;

                filled_size += fill.size;
                buffer.records.push_back(record);
            }

            // Update impact with total filled size
            if (filled_size > 0) {
                impact.update(order.side, order.ts, filled_size);
            }

            // If partial, add a record for the resting limit order (same sequence)
            // Resting order is on the opposite side (buy rests as bid, sell rests as ask)
            if (order.partial) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Add";
                record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                record.price = order.price;
                record.volume = order.size - filled_size;
                record.rejected = false;
                record.partial = true;
                record.source = SOURCE_QR;

                buffer.records.push_back(record);
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            record.source = SOURCE_QR;

            buffer.records.push_back(record);
        }
    }

    return buffer;
}

Buffer run_with_race(OrderBook& lob, QRModel& model, MarketImpact& impact, Race& race, Alpha& alpha, int64_t duration, double alpha_scale, double theta) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    // RNG for race decisions
    std::mt19937_64 race_rng(std::random_device{}());

    while (time < duration) {
        // Sample dt and advance alpha process
        int64_t dt = model.sample_dt();
        alpha.step(dt);
        time += dt;

        if (time >= duration) break;

        // Combine alpha + impact bias
        double alpha_val = alpha.value();
        impact.step(time);
        double impact_val = impact.bias_factor();
        double total_bias = -alpha_val * alpha_scale + impact_val;
        model.bias(total_bias);

        // Check if race triggers
        if (race.should_race(alpha_val, race_rng)) {
            // Generate racing orders
            std::vector<Order> racers = race.generate_racers(
                alpha_val, time, lob.best_bid(), lob.best_ask(), race_rng);

            // Process each racer
            for (auto& racer : racers) {
                EventRecord base_record;
                base_record.record_lob(lob);
                base_record.bias = total_bias;
                base_record.alpha = alpha_val;

                if (racer.type == OrderType::Trade) {
                    fills.clear();
                    lob.process(racer, &fills);

                    int64_t order_seq = seq++;
                    int32_t filled_size = 0;

                    for (const auto& fill : fills) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = fill.price;
                        record.volume = fill.size;
                        record.rejected = false;
                        record.partial = false;
                        record.source = SOURCE_RACE;

                        filled_size += fill.size;
                        buffer.records.push_back(record);
                    }

                    // Update impact with total filled size
                    if (filled_size > 0) {
                        impact.update(racer.side, racer.ts, filled_size);
                    }

                    if (racer.partial) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Add";
                        record.side = (racer.side == Side::Bid) ? Side::Ask : Side::Bid;
                        record.price = racer.price;
                        record.volume = racer.size - filled_size;
                        record.rejected = false;
                        record.partial = true;
                        record.source = SOURCE_RACE;

                        buffer.records.push_back(record);
                    }
                } else {
                    // Cancel order
                    lob.process(racer);

                    EventRecord record = base_record;
                    record.sequence = seq++;
                    record.record_order(racer);
                    record.source = SOURCE_RACE;

                    buffer.records.push_back(record);
                }

                // Update time to racer timestamp for next iteration
                time = racer.ts;
            }

            // Consume alpha after race (information acted upon)
            if (theta > 0.0) {
                alpha.consume(theta);
            }
        }

        // Sample and process normal QR event
        Order order = model.sample_order(time);

        EventRecord base_record;
        base_record.record_lob(lob);
        base_record.bias = total_bias;
        base_record.alpha = alpha_val;

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;

            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Trade";
                record.side = order.side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_QR;

                filled_size += fill.size;
                buffer.records.push_back(record);
            }

            if (filled_size > 0) {
                impact.update(order.side, order.ts, filled_size);
            }

            if (order.partial) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Add";
                record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                record.price = order.price;
                record.volume = order.size - filled_size;
                record.rejected = false;
                record.partial = true;
                record.source = SOURCE_QR;

                buffer.records.push_back(record);
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            record.source = SOURCE_QR;

            buffer.records.push_back(record);
        }
    }

    return buffer;
}

// ============================================================================
// Unified simulation function (replaces run_simple, run_with_alpha, run_with_race)
// ============================================================================
Buffer run_simulation(OrderBook& lob, QRModel& model, int64_t duration,
                      Alpha* alpha, MarketImpact* impact,
                      Race* race) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    // RNG for race decisions (only used if race != nullptr)
    std::mt19937_64 race_rng(std::random_device{}());

    while (time < duration) {
        // Get current alpha value for race decision
        double alpha_val = alpha ? alpha->value() : 0.0;

        // Check if race should trigger (only if race mechanism is enabled)
        if (race && race->should_race(alpha_val, race_rng)) {
            // === RACE PATH ===
            int64_t race_start = time;

            // Generate racing orders with timestamps
            std::vector<Order> racers = race->generate_racer_orders(
                alpha_val, lob.best_bid(), lob.best_ask(), race_rng);
            race->assign_timestamps(racers, race_start, race_rng);

            // Process each racer
            for (auto& racer : racers) {
                EventRecord base_record;
                base_record.record_lob(lob);
                base_record.alpha = alpha_val;

                // Update time to racer timestamp
                time = racer.ts;
								if (impact) {
										impact->step(time);
								}

                if (racer.type == OrderType::Trade) {
                    fills.clear();
                    lob.process(racer, &fills);

                    int64_t order_seq = seq++;
                    int32_t filled_size = 0;

                    for (const auto& fill : fills) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = fill.price;
                        record.volume = fill.size;
                        record.rejected = false;
                        record.partial = false;
                        record.source = SOURCE_RACE;

                        filled_size += fill.size;
                        buffer.records.push_back(record);
                    }

                    // Update impact with total filled size
                    if (impact && filled_size > 0) {
                        impact->add_trade(racer.side, filled_size);
                    }

                    if (racer.rejected) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = racer.price;
                        record.volume = racer.size;
                        record.rejected = true;
                        record.partial = false;
                        record.source = SOURCE_RACE;
                        buffer.records.push_back(record);
                    }

                    if (racer.partial) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Add";
                        record.side = (racer.side == Side::Bid) ? Side::Ask : Side::Bid;
                        record.price = racer.price;
                        record.volume = racer.size - filled_size;
                        record.rejected = false;
                        record.partial = true;
                        record.source = SOURCE_RACE;
                        buffer.records.push_back(record);
                    }
                } else {
                    // Cancel or Add order
                    lob.process(racer);

                    EventRecord record = base_record;
                    record.sequence = seq++;
                    record.record_order(racer);
                    record.source = SOURCE_RACE;
                    buffer.records.push_back(record);
                }
            }

            // Advance alpha for race duration
            if (alpha) {
                alpha->step(time - race_start);
            }

            // Consume alpha after race (information acted upon)
            if (alpha && race->alpha_decay() > 0.0) {
                alpha->consume(race->alpha_decay());
            }
						
        }
				
				else {
						int64_t dt = model.sample_dt();
						time += dt;
						if (time >= duration) break;

						alpha_val = alpha ? alpha->value() : 0.0;
						double impact_val = impact ? impact->bias_factor() : 0.0;
						double total_bias = -alpha_val + impact_val;
						model.bias(total_bias);
						Order order = model.sample_order(time);

						if (alpha) {
								alpha->step(dt);
						}
						if (impact) {
								impact->step(time);
						}

						// Capture LOB state before processing
						EventRecord base_record;
						base_record.record_lob(lob);
						base_record.bias = total_bias;
						base_record.alpha = alpha_val;

						if (order.type == OrderType::Trade) {
								fills.clear();
								lob.process(order, &fills);

								int64_t order_seq = seq++;
								int32_t filled_size = 0;

								// Create a record for each fill
								for (const auto& fill : fills) {
										EventRecord record = base_record;
										record.sequence = order_seq;
										record.timestamp = order.ts;
										record.type = "Trade";
										record.side = order.side;
										record.price = fill.price;
										record.volume = fill.size;
										record.rejected = false;
										record.partial = false;
										record.source = SOURCE_QR;

										filled_size += fill.size;
										buffer.records.push_back(record);
								}

								// Update impact with total filled size
								if (impact && filled_size > 0) {
										impact->add_trade(order.side, filled_size);
								}

								// If partial, add a record for the resting limit order
								if (order.partial) {
										EventRecord record = base_record;
										record.sequence = order_seq;
										record.timestamp = order.ts;
										record.type = "Add";
										record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
										record.price = order.price;
										record.volume = order.size - filled_size;
										record.rejected = false;
										record.partial = true;
										record.source = SOURCE_QR;
										buffer.records.push_back(record);
								}
						} else {
								lob.process(order);

								EventRecord record = base_record;
								record.sequence = seq++;
								record.record_order(order);
								record.source = SOURCE_QR;
								buffer.records.push_back(record);
						}
				}
    }

    return buffer;
}

// ============================================================================
// Strategy simulation with race mechanism
// ============================================================================
std::pair<Buffer, StrategyBuffer> run_aggressive(OrderBook& lob, QRModel& model,
                                                  MarketImpact& impact, Race* race,
                                                  Alpha& alpha, AggressiveStrategy& strategy,
                                                  int64_t duration, double alpha_scale, double theta) {
    Buffer buffer;
    StrategyBuffer strategy_buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    // Cooldown tracking: events since last strategy trade
    int32_t events_since_last_trade = strategy.cooldown();  // Start ready to trade
    int32_t cooldown = strategy.cooldown();

    // RNG for race decisions
    std::mt19937_64 race_rng(std::random_device{}());

    while (time < duration) {
        // Get current alpha value for race/strategy decisions
        double alpha_val = alpha.value();

        // Check if strategy wants to trade (only if cooldown elapsed)
        std::optional<Order> strat_order;
        if (events_since_last_trade >= cooldown) {
            strat_order = strategy.decide(alpha_val, time, lob);
        }

        // Check if race triggers (only if race mechanism enabled)
        bool race_triggers = race && race->should_race(alpha_val, race_rng);

        if (race_triggers) {
            // === RACE PATH ===
            int64_t race_start = time;

            // Generate racer orders WITHOUT timestamps
            std::vector<Order> racers = race->generate_racer_orders(
                alpha_val, lob.best_bid(), lob.best_ask(), race_rng);

            // Insert strategy order at random position if strategy wants to trade
            if (strat_order) {
                strat_order->order_id = -1;  // Mark as strategy order
                std::uniform_int_distribution<size_t> pos_dist(0, racers.size());
                size_t insert_pos = pos_dist(race_rng);
                racers.insert(racers.begin() + insert_pos, *strat_order);
            }

            // Assign timestamps to ALL orders (delta for first, gamma for rest)
            race->assign_timestamps(racers, race_start, race_rng);

            // Process each racer
            for (auto& racer : racers) {
                EventRecord base_record;
                base_record.record_lob(lob);
                base_record.alpha = alpha_val;

                // Determine source: strategy or race
                int8_t order_source = (racer.order_id == -1) ? SOURCE_STRATEGY : SOURCE_RACE;

                if (racer.type == OrderType::Trade) {
                    fills.clear();
                    lob.process(racer, &fills);

                    int64_t order_seq = seq++;
                    int32_t filled_size = 0;

                    // If this is the strategy order
                    if (racer.order_id == -1) {
                        for (const auto& fill : fills) {
                            filled_size += fill.size;
                        }
                        strategy.on_fill(racer, filled_size, racer.rejected);

                        StrategyTradeRecord strat_rec;
                        strat_rec.timestamp = racer.ts;
                        strat_rec.inventory = strategy.inventory();
                        strat_rec.side = racer.side;
                        strat_rec.size = racer.size;
                        strat_rec.price = racer.price;
                        strat_rec.rejected = racer.rejected;
                        strat_rec.pnl = strategy.pnl();
                        strategy_buffer.trades.push_back(strat_rec);

                        // Reset cooldown
                        events_since_last_trade = 0;
                    } else {
                        for (const auto& fill : fills) {
                            filled_size += fill.size;
                        }
                        // Count non-rejected event for cooldown
                        if (!racer.rejected) {
                            events_since_last_trade++;
                        }
                    }

                    for (const auto& fill : fills) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = fill.price;
                        record.volume = fill.size;
                        record.rejected = false;
                        record.partial = false;
                        record.source = order_source;
                        buffer.records.push_back(record);
                    }

                    // Update impact with total filled size
                    if (filled_size > 0) {
                        impact.update(racer.side, racer.ts, filled_size);
                    }

                    if (racer.rejected) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = racer.price;
                        record.volume = racer.size;
                        record.rejected = true;
                        record.partial = false;
                        record.source = order_source;
                        buffer.records.push_back(record);
                    }

                    if (racer.partial) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Add";
                        record.side = (racer.side == Side::Bid) ? Side::Ask : Side::Bid;
                        record.price = racer.price;
                        record.volume = racer.size - filled_size;
                        record.rejected = false;
                        record.partial = true;
                        record.source = order_source;
                        buffer.records.push_back(record);
                    }
                } else {
                    // Cancel or Add order
                    lob.process(racer);

                    EventRecord record = base_record;
                    record.sequence = seq++;
                    record.record_order(racer);
                    record.source = order_source;
                    buffer.records.push_back(record);

                    // Count non-rejected event for cooldown (if not strategy order)
                    if (racer.order_id != -1 && !racer.rejected) {
                        events_since_last_trade++;
                    }
                }

                // Update time to racer timestamp
                time = racer.ts;
            }

            // Advance alpha for race duration
            alpha.step(time - race_start);

            // Consume alpha after race
            if (theta > 0.0) {
                alpha.consume(theta);
            }
        } else if (strat_order) {
            // === STRATEGY TRADES DIRECTLY (no race) ===
            EventRecord base_record;
            base_record.record_lob(lob);
            base_record.alpha = alpha_val;

            fills.clear();
            lob.process(*strat_order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;
            for (const auto& fill : fills) {
                filled_size += fill.size;
            }

            if (filled_size > 0) {
                impact.update(strat_order->side, strat_order->ts, filled_size);
            }

            strategy.on_fill(*strat_order, filled_size, strat_order->rejected);

            StrategyTradeRecord strat_rec;
            strat_rec.timestamp = strat_order->ts;
            strat_rec.inventory = strategy.inventory();
            strat_rec.side = strat_order->side;
            strat_rec.size = strat_order->size;
            strat_rec.price = strat_order->price;
            strat_rec.rejected = strat_order->rejected;
            strat_rec.pnl = strategy.pnl();
            strategy_buffer.trades.push_back(strat_rec);

            // Record fills
            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = strat_order->ts;
                record.type = "Trade";
                record.side = strat_order->side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_STRATEGY;
                buffer.records.push_back(record);
            }

            if (strat_order->rejected) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = strat_order->ts;
                record.type = "Trade";
                record.side = strat_order->side;
                record.price = strat_order->price;
                record.volume = strat_order->size;
                record.rejected = true;
                record.partial = false;
                record.source = SOURCE_STRATEGY;
                buffer.records.push_back(record);
            }

            // Reset cooldown
            events_since_last_trade = 0;

            // Sample round-trip delay if race mechanism is available
            if (race) {
                int64_t roundtrip = race->sample_roundtrip(race_rng);
                time += roundtrip;
            }
        }

        // Sample dt and advance alpha process
        int64_t dt = model.sample_dt();
        alpha.step(dt);
        time += dt;

        if (time >= duration) break;

        // Set bias with current alpha (after all evolution)
        alpha_val = alpha.value();
        impact.step(time);
        double impact_val = impact.bias_factor();
        double total_bias = -alpha_val * alpha_scale + impact_val;
        model.bias(total_bias);

        // Sample and process normal QR event
        Order order = model.sample_order(time);

        EventRecord base_record;
        base_record.record_lob(lob);
        base_record.bias = total_bias;
        base_record.alpha = alpha_val;

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;

            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Trade";
                record.side = order.side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_QR;

                filled_size += fill.size;
                buffer.records.push_back(record);
            }

            if (filled_size > 0) {
                impact.update(order.side, order.ts, filled_size);
            }

            if (order.partial) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Add";
                record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                record.price = order.price;
                record.volume = order.size - filled_size;
                record.rejected = false;
                record.partial = true;
                record.source = SOURCE_QR;
                buffer.records.push_back(record);
            }

            // Count non-rejected event for cooldown
            if (!order.rejected) {
                events_since_last_trade++;
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            record.source = SOURCE_QR;
            buffer.records.push_back(record);

            // Count non-rejected event for cooldown
            if (!order.rejected) {
                events_since_last_trade++;
            }
        }
    }

    return {buffer, strategy_buffer};
}

// ============================================================================
// HFT Alpha simulation: races without forced QR events after each race
// Alpha fades naturally through imbalance mechanism
// ============================================================================
Buffer run_hft_alpha(OrderBook& lob, QRModel& model, int64_t duration,
                     Alpha& alpha, Race& race, double w_ou,
                     double w_imb, double alpha_scale,
                     MarketImpact* impact) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;
    std::vector<Fill> fills;

    // RNG for race decisions
    std::mt19937_64 race_rng(std::random_device{}());

    while (time < duration) {
        // Sample dt and advance alpha process
        int64_t dt = model.sample_dt();
        alpha.step(dt);
        time += dt;

        if (time >= duration) break;

        // Combined alpha signal: w_ou * X_ou + w_imb * imbalance
        double alpha_val = w_ou * alpha.value() + w_imb * lob.imbalance();

        // Set bias with current alpha + optional impact
        if (impact) impact->step(time);
        double impact_val = impact ? impact->bias_factor() : 0.0;
        double total_bias = -alpha_val * alpha_scale + impact_val;
        model.bias(total_bias);

        // Check if race should trigger
        if (race.should_race(alpha_val, race_rng)) {
            // === RACE PATH ===
            int64_t race_start = time;

            // Generate racing orders with timestamps
            std::vector<Order> racers = race.generate_racer_orders(
                alpha_val, lob.best_bid(), lob.best_ask(), race_rng);
            race.assign_timestamps(racers, race_start, race_rng);

            // Process each racer
            for (auto& racer : racers) {
                EventRecord base_record;
                base_record.record_lob(lob);
                base_record.bias = total_bias;
                base_record.alpha = alpha_val;

                if (racer.type == OrderType::Trade) {
                    fills.clear();
                    lob.process(racer, &fills);

                    int64_t order_seq = seq++;
                    int32_t filled_size = 0;

                    for (const auto& fill : fills) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = fill.price;
                        record.volume = fill.size;
                        record.rejected = false;
                        record.partial = false;
                        record.source = SOURCE_RACE;

                        filled_size += fill.size;
                        buffer.records.push_back(record);
                    }

                    // Update impact with total filled size
                    if (impact && filled_size > 0) {
                        impact->update(racer.side, racer.ts, filled_size);
                    }

                    if (racer.rejected) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Trade";
                        record.side = racer.side;
                        record.price = racer.price;
                        record.volume = racer.size;
                        record.rejected = true;
                        record.partial = false;
                        record.source = SOURCE_RACE;
                        buffer.records.push_back(record);
                    }

                    if (racer.partial) {
                        EventRecord record = base_record;
                        record.sequence = order_seq;
                        record.timestamp = racer.ts;
                        record.type = "Add";
                        record.side = (racer.side == Side::Bid) ? Side::Ask : Side::Bid;
                        record.price = racer.price;
                        record.volume = racer.size - filled_size;
                        record.rejected = false;
                        record.partial = true;
                        record.source = SOURCE_RACE;
                        buffer.records.push_back(record);
                    }
                } else {
                    // Cancel or Add order
                    lob.process(racer);

                    EventRecord record = base_record;
                    record.sequence = seq++;
                    record.record_order(racer);
                    record.source = SOURCE_RACE;
                    buffer.records.push_back(record);
                }

                // Update time to racer timestamp
                time = racer.ts;
            }

            // Advance alpha for race duration
            alpha.step(time - race_start);

            // No forced QR event after race - just continue to next iteration
            continue;
        }

        // === QR PATH (no race triggered) ===
        Order order = model.sample_order(time);

        // Capture LOB state before processing
        EventRecord base_record;
        base_record.record_lob(lob);
        base_record.bias = total_bias;
        base_record.alpha = alpha_val;

        if (order.type == OrderType::Trade) {
            fills.clear();
            lob.process(order, &fills);

            int64_t order_seq = seq++;
            int32_t filled_size = 0;

            // Create a record for each fill
            for (const auto& fill : fills) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Trade";
                record.side = order.side;
                record.price = fill.price;
                record.volume = fill.size;
                record.rejected = false;
                record.partial = false;
                record.source = SOURCE_QR;

                filled_size += fill.size;
                buffer.records.push_back(record);
            }

            // Update impact with total filled size
            if (impact && filled_size > 0) {
                impact->update(order.side, time, filled_size);
            }

            // If partial, add a record for the resting limit order
            if (order.partial) {
                EventRecord record = base_record;
                record.sequence = order_seq;
                record.timestamp = order.ts;
                record.type = "Add";
                record.side = (order.side == Side::Bid) ? Side::Ask : Side::Bid;
                record.price = order.price;
                record.volume = order.size - filled_size;
                record.rejected = false;
                record.partial = true;
                record.source = SOURCE_QR;
                buffer.records.push_back(record);
            }
        } else {
            lob.process(order);

            EventRecord record = base_record;
            record.sequence = seq++;
            record.record_order(order);
            record.source = SOURCE_QR;
            buffer.records.push_back(record);
        }
    }

    return buffer;
}

}
