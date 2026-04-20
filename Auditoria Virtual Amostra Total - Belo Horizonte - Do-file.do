clear

use "C:\Users\amand\Amanda\GitHub\gemini_built_environment\Auditoria Virtual Amostra Total - Belo Horizonte.dta" 

keep identificador data_imagens swalk_pres_cat str_cwalk str_scont1 str_cond_cat2 str_tcont3 str_tcont1 swalk_cont_cat2 str_blane veg_tree_cat2 str_med_cat2 str_lanevis_cat2 str_cstop_cat2 trans_blane trans_stop str_tcont2

ds identificador, not
foreach var in `r(varlist)' {
    rename `var' `var'_0
}

save "C:\Users\amand\Amanda\GitHub\gemini_built_environment\Auditoria Virtual Amostra Total - Belo Horizonte v2.dta", replace