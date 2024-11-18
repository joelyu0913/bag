#include <algorithm>
#include <string_view>
#include <vector>

#include "yang/data/null.h"
#include "yang/io/open.h"
#include "yang/sim/module.h"
#include "yang/util/dates.h"
#include "yang/util/simple_csv.h"
#include "yang/util/strings.h"
#include "yang/util/unordered_map.h"

namespace yang {

class CnBaseStd : public Module {
 protected:
  /*
   0 | SSE 主板    | SSE Main
   1 | SZSE 主板   | SZSE Main
   2 | SZSE 创业板 | SZSE ChiNext
   3 | SZSE 中小板 | SZSE SME
   4 | SSE 科创板  | SSE STAR
   5 | SSE CDR     | SSE CDR
   6 | BSE         | BSE Main
 */
  unordered_map<std::string, int> exchanges = {
      {"SSE Main", 0}, {"SZSE Main", 1}, {"SZSE ChiNext", 2}, {"SZSE SME", 3},
      {"SSE STAR", 4}, {"SSE CDR", 5},   {"BSE Main", 6}};

  void RunImpl() final {
    auto &env = this->env();

    unordered_map<std::string_view, int> indices;
    for (auto &idx : env.univ_indices()) {
      indices.emplace(idx, env.univ().Find(idx));
    }

    auto open_arr = WriteArray<float>("open");
    auto close_arr = WriteArray<float>("close");
    auto high_arr = WriteArray<float>("high");
    auto low_arr = WriteArray<float>("low");
    auto vol_arr = WriteArray<float>("vol", 0);
    auto dvol_arr = WriteArray<float>("dvol", 0);
    auto vwap_arr = WriteArray<float>("vwap");

    auto sector_idx = Index<std::string>::Load(cache_dir().GetPath(name(), "sector_idx"));
    auto industry_idx = Index<std::string>::Load(cache_dir().GetPath(name(), "industry_idx"));
    auto subindustry_idx = Index<std::string>::Load(cache_dir().GetPath(name(), "subindustry_idx"));
    if (sector_idx.empty()) {
      sector_idx.Insert("");
    }
    if (industry_idx.empty()) {
      industry_idx.Insert("");
    }
    if (subindustry_idx.empty()) {
      subindustry_idx.Insert("");
    }

    auto cty_arr = WriteArray<int>("cty", -1);
    auto sector_arr = WriteArray<int>("sector", -1);
    auto industry_arr = WriteArray<int>("industry", -1);
    auto subindustry_arr = WriteArray<int>("subindustry", -1);
    auto sharesout_arr = WriteArray<float>("sharesout", 0);
    auto sharesfloat_arr = WriteArray<float>("sharesfloat", 0);
    auto cap_arr = WriteArray<float>("cap", 0);
    auto cumadj_arr = WriteArray<float>("cumadj");
    auto adj_arr = WriteArray<float>("adj", 1.0);
    auto st_arr = WriteArray<int>("st", -1);
    auto exch_arr = WriteArray<int>("exch", -1);
    auto halt_arr = WriteArray<bool>("halt", false);
    auto limit_up_arr = WriteArray<float>("limit_up");
    auto limit_down_arr = WriteArray<float>("limit_down");

    auto univ_all = WriteArray<bool>("univ_all");
    auto listing = Array<bool>::MMap(cache_dir().GetPath("env", "listing"));

    auto raw_prc_file = env.config<std::string>("raw_prc_file");
    auto index_file = env.config<std::string>("index_file");
    for (int di = start_di(); di < end_di(); ++di) {
      int date = env.dates()[di];

      auto raw_prc_path = FormatDate(raw_prc_file, date);
      auto index_path = FormatDate(index_file, date);
      LOG_INFO("Loading {}", raw_prc_path);
      LOG_INFO("Loading {}", index_path);

      int updated_stock = 0;
      int updated_idx = 0;

      univ_all.row_vec(di).copy_from(listing.row_vec(di));

      try {
        auto file = io::OpenBufferedFile(raw_prc_path);
        if (file->error()) throw yang::IoError("Failed to open " + raw_prc_path);
        std::string line;
        file->ReadLine(line);
        SimpleCsvReader reader(line, '|');

        while (file->ReadLine(line)) {
          reader.ReadNext(line);
          auto sid = reader["sid"];
          if (sid.size() != 9) LOG_FATAL("sid {} error", sid);

          int ii = env.univ().Find(di, sid);

          if (ii < 0) continue;
          open_arr(di, ii) = reader.Get<float>("open");
          close_arr(di, ii) = reader.Get<float>("close");

          high_arr(di, ii) = reader.Get<float>("high");
          low_arr(di, ii) = reader.Get<float>("low");
          vol_arr(di, ii) = reader.Get<float>("vol");
          dvol_arr(di, ii) = reader.Get<float>("dvol");
          vwap_arr(di, ii) = vol_arr(di, ii) == 0 ? 0 : dvol_arr(di, ii) / vol_arr(di, ii);

          auto sector = reader["sector"];
          auto industry = reader["ind"];
          auto subindustry = reader["subind"];

          cty_arr(di, ii) = 1;
          sector_arr(di, ii) = sector_idx.Insert(std::string(sector));
          industry_arr(di, ii) = industry_idx.Insert(std::string(industry));
          subindustry_arr(di, ii) = subindustry_idx.Insert(std::string(subindustry));

          sharesout_arr(di, ii) = reader.Get<float>("sho");
          sharesfloat_arr(di, ii) = reader.Get<float>("flo");
          cap_arr(di, ii) = sharesout_arr(di, ii) * reader.Get<float>("pclose");

          adj_arr(di, ii) = reader.Get<float>("adj");
          if (!IsValid(adj_arr(di, ii))) {
            adj_arr(di, ii) = 1.;
          }

          if (di > 0) {
            float adj_ = close_arr(di - 1, ii) / reader.Get<float>("pclose");
            if (std::abs(adj_arr(di, ii) - adj_) > 1e-5) {
              LOG_ERROR("Invalid adj: {} - {}, {} - {}: {}, {}", di, dates()[di], ii, univ()[ii],
                        adj_arr(di, ii), adj_);
            }
          }
          auto sid_name = reader["name"];
          auto st_pos = sid_name.find("ST");

          if (st_pos == std::string::npos) {
            st_arr(di, ii) = 0;
          } else {
            if (st_pos > 2) LOG_WARN("{},{},{}", sid_name, st_pos, sid_name.size());
            st_arr(di, ii) = 1;
          }

          if (reader.header().count("halt")) {
            halt_arr(di, ii) = reader.Get<int>("halt", 0) != 0;
          } else {
            halt_arr(di, ii) = 1 - reader.Get<int>("active", 1);
          }
          if (halt_arr(di, ii)) univ_all(di, ii) = false;

          limit_up_arr(di, ii) = reader.Get<float>("up");
          limit_down_arr(di, ii) = reader.Get<float>("down");
          auto exch = reader["exch"];
          if (exchanges.count(exch)) {
            exch_arr(di, ii) = exchanges.at(exch);
          } else {
            LOG_WARN("Unknown exchange: {}", exch);
          }

          if (indices.contains(std::string_view(sid))) {
            updated_idx++;
          } else {
            updated_stock++;
          }
        }
      } catch (const std::exception &ex) {
        LOG_FATAL("Failed to load {}: {}", raw_prc_path, ex.what());
      }
      try {
        auto index_file = io::OpenBufferedFile(index_path);
        if (index_file->error()) throw yang::IoError("Failed to open " + index_path);
        std::string line;
        index_file->ReadLine(line);
        SimpleCsvReader reader(line, '|');

        while (index_file->ReadLine(line)) {
          reader.ReadNext(line);
          auto iid = std::string(reader["sid"]);
          if (iid.size() != 9) LOG_FATAL("sid {} error", iid);
          int ii = env.univ().Find(di, iid);

          if (ii < 0) continue;

          open_arr(di, ii) = reader.Get<float>("open");
          close_arr(di, ii) = reader.Get<float>("close");
          high_arr(di, ii) = reader.Get<float>("high");
          low_arr(di, ii) = reader.Get<float>("low");
          vol_arr(di, ii) = reader.Get<float>("vol");
          dvol_arr(di, ii) = reader.Get<float>("dvol");
          vwap_arr(di, ii) = vol_arr(di, ii) == 0 ? 0 : dvol_arr(di, ii) / vol_arr(di, ii);

          cumadj_arr(di, ii) = 1;
          adj_arr(di, ii) = 1;
          univ_all(di, ii) = true;

          limit_up_arr(di, ii) = high_arr(di, ii) * 2;
          limit_down_arr(di, ii) = 0.;

          if (indices.contains(iid)) {
            updated_idx++;
          }
        }
      } catch (const std::exception &ex) {
        LOG_FATAL("Failed to load {}: {}", index_path, ex.what());
      }
      LOG_INFO("[{}] [{}] Loaded {} stocks, {} indices", name(), date, updated_stock, updated_idx);
    }
    sector_idx.Save();
    industry_idx.Save();
    subindustry_idx.Save();

    auto ret_arr = WriteArray<float>("ret");
    auto adj_open_arr = WriteArray<float>("adj_open");
    auto adj_close_arr = WriteArray<float>("adj_close");
    auto adj_high_arr = WriteArray<float>("adj_high");
    auto adj_low_arr = WriteArray<float>("adj_low");
    auto adj_vol_arr = WriteArray<float>("adj_vol", 0);
    auto adj_vwap_arr = WriteArray<float>("adj_vwap");

    for (int ii = 0; ii < env.max_univ_size(); ++ii) {
      cumadj_arr(0, ii) = 1.;
    }
    for (int di = start_di(); di < end_di(); ++di) {
      for (int ii = 0; ii < env.max_univ_size(); ++ii) {
        if (di == 0) {
          cumadj_arr(di, ii) = 1.;
        } else {
          cumadj_arr(di, ii) = cumadj_arr(di - 1, ii) * adj_arr(di, ii);
        }
        adj_open_arr(di, ii) = open_arr(di, ii) * cumadj_arr(di, ii);
        adj_close_arr(di, ii) = close_arr(di, ii) * cumadj_arr(di, ii);
        adj_high_arr(di, ii) = high_arr(di, ii) * cumadj_arr(di, ii);
        adj_low_arr(di, ii) = low_arr(di, ii) * cumadj_arr(di, ii);
        adj_vwap_arr(di, ii) = vwap_arr(di, ii) * cumadj_arr(di, ii);

        adj_vol_arr(di, ii) = vol_arr(di, ii) / cumadj_arr(di, ii);

        if (di > 0) {
          ret_arr(di, ii) = close_arr(di, ii) * adj_arr(di, ii) / close_arr(di - 1, ii) - 1;
        } else {
          ret_arr(di, ii) = NAN;
        }
      }
    }
  }
};

REGISTER_MODULE(CnBaseStd);

}  // namespace yang
