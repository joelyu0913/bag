from basic.base_daily import BaseDaily


class Model(BaseDaily):
    def run_impl(self):
        start_datetime = self.datetimes[self.start_di]
        end_datetime = self.datetimes[self.end_di - 1]

        from prophet.predict import run as run_predict

        grf = self.config.get("grf", False)
        if grf:
            output_name = "data"
        else:
            self.base_load("sig")
            output_name = "b_sig"

        run_predict(
            workdir=self.config["model_dir"],
            config=self.config["predict_config"],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            user_data_dir=self.cache_dir.user_dir,
            sys_data_dir=self.cache_dir.sys_dir,
            output=self.cache_dir.get_path(self.name, output_name),
            update_output=True,
            inds=self.config.get("inds_path", ""),
            no_gpu=True,
            grf=grf,
            univ_mask=self.config.get("univ", ""),
        )
