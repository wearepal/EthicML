"""
Class to describe features of the SQF dataset
"""

from ethicml.data.dataset import Dataset


class Sqf(Dataset):
    """Stop, question and frisk dataset"""

    def __init__(self, split: str = "Sex", discrete_only: bool = False):
        super().__init__()
        self.discrete_only = discrete_only
        self.features = [
            "perstop",
            "sex",
            "race",
            "age",
            "ht_feet",
            "ht_inch",
            "weight",
            "weapon",
            "typeofid_O",
            "typeofid_P",
            "typeofid_R",
            "typeofid_V",
            "explnstp_N",
            "explnstp_Y",
            "othpers_N",
            "othpers_Y",
            "arstmade_N",
            "arstmade_Y",
            "offunif_N",
            "offunif_Y",
            "frisked_N",
            "frisked_Y",
            "searched_N",
            "searched_Y",
            "pf-hands_N",
            "pf-hands_Y",
            "pf-wall_N",
            "pf-wall_Y",
            "pf-grnd_N",
            "pf-grnd_Y",
            "pf-drwep_N",
            "pf-drwep_Y",
            "pf-ptwep_N",
            "pf-ptwep_Y",
            "pf-baton_N",
            "pf-baton_Y",
            "pf-hcuff_N",
            "pf-hcuff_Y",
            "pf-pepsp_N",
            "pf-pepsp_Y",
            "pf-other_N",
            "pf-other_Y",
            "radio_N",
            "radio_Y",
            "ac-rept_N",
            "ac-rept_Y",
            "ac-inves_N",
            "ac-inves_Y",
            "rf-vcrim_N",
            "rf-vcrim_Y",
            "rf-othsw_N",
            "rf-othsw_Y",
            "ac-proxm_N",
            "ac-proxm_Y",
            "rf-attir_N",
            "rf-attir_Y",
            "cs-objcs_N",
            "cs-objcs_Y",
            "cs-descr_N",
            "cs-descr_Y",
            "cs-casng_N",
            "cs-casng_Y",
            "cs-lkout_N",
            "cs-lkout_Y",
            "rf-vcact_N",
            "rf-vcact_Y",
            "cs-cloth_N",
            "cs-cloth_Y",
            "cs-drgtr_N",
            "cs-drgtr_Y",
            "ac-evasv_N",
            "ac-evasv_Y",
            "ac-assoc_N",
            "ac-assoc_Y",
            "cs-furtv_N",
            "cs-furtv_Y",
            "rf-rfcmp_N",
            "rf-rfcmp_Y",
            "ac-cgdir_N",
            "ac-cgdir_Y",
            "rf-verbl_N",
            "rf-verbl_Y",
            "cs-vcrim_N",
            "cs-vcrim_Y",
            "cs-bulge_N",
            "cs-bulge_Y",
            "cs-other_N",
            "cs-other_Y",
            "ac-incid_N",
            "ac-incid_Y",
            "ac-time_N",
            "ac-time_Y",
            "rf-knowl_N",
            "rf-knowl_Y",
            "ac-stsnd_N",
            "ac-stsnd_Y",
            "ac-other_N",
            "ac-other_Y",
            "sb-hdobj_N",
            "sb-hdobj_Y",
            "sb-outln_N",
            "sb-outln_Y",
            "sb-admis_N",
            "sb-admis_Y",
            "sb-other_N",
            "sb-other_Y",
            "rf-furt_N",
            "rf-furt_Y",
            "rf-bulg_N",
            "rf-bulg_Y",
            "offverb_ ",
            "offverb_V",
            "offshld_ ",
            "offshld_S",
            "forceuse_ ",
            "forceuse_DO",
            "forceuse_DS",
            "forceuse_OR",
            "forceuse_OT",
            "forceuse_SF",
            "forceuse_SW",
            "haircolr_BA",
            "haircolr_BK",
            "haircolr_BL",
            "haircolr_BR",
            "haircolr_DY",
            "haircolr_GY",
            "haircolr_RA",
            "haircolr_SN",
            "haircolr_SP",
            "haircolr_XX",
            "haircolr_ZZ",
            "eyecolor_BK",
            "eyecolor_BL",
            "eyecolor_BR",
            "eyecolor_DF",
            "eyecolor_GR",
            "eyecolor_GY",
            "eyecolor_HA",
            "eyecolor_XX",
            "eyecolor_Z",
            "build_H",
            "build_M",
            "build_T",
            "build_U",
            "build_Z",
        ]

        self.continuous_features = ["perstop", "ht_feet", "age", "ht_inch", "perobs", "weight"]

        if split == "Sex":
            self.sens_attrs = ["sex"]
            self.s_prefix = ["sex"]
            self.class_labels = ["weapon"]
            self.class_label_prefix = ["weapon"]
        elif split == "Race":
            self.sens_attrs = ["race"]
            self.s_prefix = ["race"]
            self.class_labels = ["weapon"]
            self.class_label_prefix = ["weapon"]
        elif split == "Race-Sex":
            self.sens_attrs = ["sex", "race"]
            self.s_prefix = ["race", "sex"]
            self.class_labels = ["weapon"]
            self.class_label_prefix = ["weapon"]
        else:
            raise NotImplementedError

    @property
    def name(self) -> str:
        return "SQF"

    @property
    def filename(self) -> str:
        return "sqf.csv"
