#include "simulation.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

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
    arrow::Int64Builder timestamp_builder;
    arrow::StringBuilder type_builder;
    arrow::Int8Builder side_builder;
    arrow::Int32Builder price_builder;
    arrow::Int32Builder volume_builder;
    arrow::BooleanBuilder rejected_builder;
    arrow::BooleanBuilder partial_builder;
    arrow::DoubleBuilder bias_builder;
    arrow::DoubleBuilder alpha_builder;

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
        (void)timestamp_builder.Append(r.timestamp);
        (void)type_builder.Append(r.type);
        (void)side_builder.Append(static_cast<int8_t>(r.side));
        (void)price_builder.Append(r.price);
        (void)volume_builder.Append(r.volume);
        (void)rejected_builder.Append(r.rejected);
        (void)partial_builder.Append(r.partial);
        (void)bias_builder.Append(r.bias);
        (void)alpha_builder.Append(r.alpha);
    }

    std::shared_ptr<arrow::Array> sequence_arr;
    std::shared_ptr<arrow::Array> best_bid_price_arr, best_bid_vol_arr;
    std::shared_ptr<arrow::Array> best_ask_price_arr, best_ask_vol_arr;
    std::shared_ptr<arrow::Array> second_bid_price_arr, second_bid_vol_arr;
    std::shared_ptr<arrow::Array> second_ask_price_arr, second_ask_vol_arr;
    std::shared_ptr<arrow::Array> timestamp_arr, type_arr, side_arr;
    std::shared_ptr<arrow::Array> price_arr, volume_arr, rejected_arr, partial_arr;
    std::shared_ptr<arrow::Array> bias_arr, alpha_arr;

    (void)sequence_builder.Finish(&sequence_arr);
    (void)best_bid_price_builder.Finish(&best_bid_price_arr);
    (void)best_bid_vol_builder.Finish(&best_bid_vol_arr);
    (void)best_ask_price_builder.Finish(&best_ask_price_arr);
    (void)best_ask_vol_builder.Finish(&best_ask_vol_arr);
    (void)second_bid_price_builder.Finish(&second_bid_price_arr);
    (void)second_bid_vol_builder.Finish(&second_bid_vol_arr);
    (void)second_ask_price_builder.Finish(&second_ask_price_arr);
    (void)second_ask_vol_builder.Finish(&second_ask_vol_arr);
    (void)timestamp_builder.Finish(&timestamp_arr);
    (void)type_builder.Finish(&type_arr);
    (void)side_builder.Finish(&side_arr);
    (void)price_builder.Finish(&price_arr);
    (void)volume_builder.Finish(&volume_arr);
    (void)rejected_builder.Finish(&rejected_arr);
    (void)partial_builder.Finish(&partial_arr);
    (void)bias_builder.Finish(&bias_arr);
    (void)alpha_builder.Finish(&alpha_arr);

    auto schema = arrow::schema({
        arrow::field("sequence", arrow::int64()),
        arrow::field("best_bid_price", arrow::int32()),
        arrow::field("best_bid_vol", arrow::int32()),
        arrow::field("best_ask_price", arrow::int32()),
        arrow::field("best_ask_vol", arrow::int32()),
        arrow::field("second_bid_price", arrow::int32()),
        arrow::field("second_bid_vol", arrow::int32()),
        arrow::field("second_ask_price", arrow::int32()),
        arrow::field("second_ask_vol", arrow::int32()),
        arrow::field("timestamp", arrow::int64()),
        arrow::field("type", arrow::utf8()),
        arrow::field("side", arrow::int8()),
        arrow::field("price", arrow::int32()),
        arrow::field("volume", arrow::int32()),
        arrow::field("rejected", arrow::boolean()),
        arrow::field("partial", arrow::boolean()),
        arrow::field("bias", arrow::float64()),
        arrow::field("alpha", arrow::float64())
    });

    auto table = arrow::Table::Make(schema, {
        sequence_arr,
        best_bid_price_arr, best_bid_vol_arr,
        best_ask_price_arr, best_ask_vol_arr,
        second_bid_price_arr, second_bid_vol_arr,
        second_ask_price_arr, second_ask_vol_arr,
        timestamp_arr, type_arr, side_arr,
        price_arr, volume_arr, rejected_arr, partial_arr,
        bias_arr, alpha_arr
    });

    auto outfile = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    (void)parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, records.size());
}

Buffer run_simple(OrderBook& lob, QRModel& model, int64_t duration) {
    Buffer buffer;
    int64_t time = 0;
    int64_t seq = 0;

    while (time < duration) {
        int64_t dt = model.sample_dt();
        time += dt;

        if (time >= duration) break;

        model.bias(0.0);
        Order order = model.sample_order(time);

        EventRecord record;
        record.sequence = seq++;
        record.record_lob(lob);
        lob.process(order);
        record.record_order(order);

        buffer.records.push_back(record);
    }

    return buffer;
}

Buffer run_metaorder(OrderBook& lob, QRModel& model, MarketImpact& impact, MetaOrder& metaorder, int64_t duration) {
    Buffer buffer;
    int64_t time = 0;
    size_t i = 0;
    size_t n = metaorder.timestamps.size();
    double current_bias = 0.0;
    double sign_sum = 0.0;
    int trade_count = 0;
    double current_sign_mean = 0.0;

    Order order;
    while (time < duration) {
        current_bias = impact.bias_factor(time);
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
            impact.update(order.side, time);
            double sign = (order.side == Side::Bid) ? -1.0 : 1.0;
            sign_sum += sign;
            trade_count++;
            current_sign_mean = sign_sum / trade_count;
        }
        record.trade_sign_mean = current_sign_mean;

        buffer.records.push_back(record);
    }
    return buffer;
}

Buffer run_with_alpha(OrderBook& lob, QRModel& model, MarketImpact& impact, Alpha& alpha, int64_t duration) {
    Buffer buffer;
    int64_t time = 0;
    double sign_sum = 0.0;
    int trade_count = 0;
    double current_sign_mean = 0.0;

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
        double impact_val = impact.bias_factor(time);
        double total_bias = -alpha_val + impact_val;
				// double total_bias = impact_val;
        model.bias(total_bias);

        Order order = model.sample_order(time);

        EventRecord record;
        record.record_lob(lob);
        lob.process(order);
        record.record_order(order);
        record.bias = total_bias;
        record.alpha = alpha_val;

        if (order.type == OrderType::Trade) {
            impact.update(order.side, time);
            double sign = (order.side == Side::Bid) ? -1.0 : 1.0;
            sign_sum += sign;
            trade_count++;
            current_sign_mean = sign_sum / trade_count;
        }
        record.trade_sign_mean = current_sign_mean;

        buffer.records.push_back(record);
    }

    return buffer;
}

}
