clear

* Import CSV file
import delimited "C:\Users\amand\Amanda\GitHub\gemini_built_environment\streetview_predictions_2.5-flash.csv"

* Save as Stata dataset
save "C:\Users\amand\Amanda\GitHub\gemini_built_environment\streetview_predictions_2.5-flash.dta", replace


* Keep only valid observations
keep if status == "ok"

* Standardize ID variable
rename id identificador
format %20.0g identificador

* ---- Date handling ----
rename pano_date data_imagens
gen data_imagens_m = monthly(data_imagens, "YM")
format data_imagens_m %tm

* ---- Rename variables to standardized names ----
rename sidewalks swalk_pres_cat
rename crosswalks str_cwalk
rename speed_bumps str_scont1
rename pothole str_cond_cat2
rename traffic_light str_tcont3
rename stop_sign str_tcont1
rename sidewalk_obstruction swalk_cont_cat2
rename bike_lane str_blane
rename trees veg_tree_cat2
rename median str_med_cat2
rename lane_markings str_lanevis_cat2
rename crossing_sign str_cstop_cat2
rename bus_lane trans_blane
rename bus_stop trans_stop
rename roundabout str_tcont2

duplicates report identificador

* ---- Collapse ----
collapse (max) data_imagens_m ///
               swalk_pres_cat str_cwalk str_scont1 str_cond_cat2 ///
               str_tcont3 str_tcont1 swalk_cont_cat2 str_blane ///
               veg_tree_cat2 str_med_cat2 str_lanevis_cat2 ///
               str_cstop_cat2 trans_blane trans_stop str_tcont2, ///
               by(identificador)

* ---- Convert date ----
gen data_imagens_d = dofm(data_imagens_m)
format data_imagens_d %tdCCYY-NN-DD

drop data_imagens_m
rename data_imagens_d data_imagens

duplicates report identificador

move data_imagens identificador
move data_imagens identificador

* ---- Add suffix _1 ----
ds identificador, not
foreach var in `r(varlist)' {
    rename `var' `var'_1
}

save "C:\Users\amand\Amanda\GitHub\gemini_built_environment\streetview_predictions_2.5-flash v2.dta", replace


* ---- Merge ----
merge 1:1 identificador using "C:\Users\amand\Amanda\GitHub\gemini_built_environment\Auditoria Virtual Amostra Total - Belo Horizonte v2.dta"

drop _merge

save "C:\Users\amand\Amanda\GitHub\gemini_built_environment\streetview_predictions_2.5-flash v3.dta", replace