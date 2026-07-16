import os

from mods import ExpACCV2026

EXP_FLOAT_VALUES = {"cls_feat", "train_subset_pct", "train_min_cls_pct"}


def check_exp_value(exp):
    for value in EXP_FLOAT_VALUES:
        if getattr(exp, value, None) is not None:
            setattr(exp, value, float(getattr(exp, value)))


class Exp(ExpACCV2026):
    def __init__(self):
        super().__init__()
        self.cls_feat_dim = 320
        self.cls_feat = 0
        self.cls_feat_loss = "sup_con_loss"
        self.cls_feat_temperature = 0.07

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = f"{self.exp_name}_baseline"

        # ---------------- dataloader config ---------------- #

        # Define yourself dataset path
        self.data_dir = "datasets/UNIMIB2016"
        self.train_ann = "annotation_train.json"
        self.val_ann = "annotation_test.json"

        self.train_subset_pct = None
        self.train_min_cls_pct = None

        # --------------  training config --------------------- #

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        # ---------------- semmel config ---------------- #

        self.num_classes = 74
        self.names = {
            0: 'food',
            1: 'arancia',
            2: 'arrosto',
            3: 'arrosto_di_vitello',
            4: 'banane',
            5: 'bruscitt',
            6: 'budino',
            7: 'carote',
            8: 'cavolfiore',
            9: 'cibo_bianco_non_identificato',
            10: 'cotoletta',
            11: 'crema_zucca_e_fagioli',
            12: 'fagiolini',
            13: 'finocchi_gratinati',
            14: 'finocchi_in_umido',
            15: 'focaccia_bianca',
            16: 'food',
            17: 'guazzetto_di_calamari',
            18: 'insalata_mista',
            19: 'lasagna_alla_bolognese',
            20: 'mandarini',
            21: 'medaglioni_di_carne',
            22: 'mele',
            23: 'merluzzo_alle_olive',
            24: 'minestra',
            25: 'minestra_lombarda',
            26: 'orecchiette_-ragu-',
            27: 'pane',
            28: 'passato_alla_piemontese',
            29: 'pasta_bianco',
            30: 'pasta_cozze_e_vongole',
            31: 'pasta_e_ceci',
            32: 'pasta_e_fagioli',
            33: 'pasta_mare_e_monti',
            34: 'pasta_pancetta_e_zucchine',
            35: 'pasta_pesto_besciamella_e_cornetti',
            36: 'pasta_ricotta_e_salsiccia',
            37: 'pasta_sugo',
            38: 'pasta_sugo_pesce',
            39: 'pasta_sugo_vegetariano',
            40: 'pasta_tonno',
            41: 'pasta_tonno_e_piselli',
            42: 'pasta_zafferano_e_piselli',
            43: 'patate-pure',
            44: 'patate-pure_prosciutto',
            45: 'patatine_fritte',
            46: 'pere',
            47: 'pesce_-filetto-',
            48: 'pesce_2_-filetto-',
            49: 'piselli',
            50: 'pizza',
            51: 'pizzoccheri',
            52: 'polpette_di_carne',
            53: 'riso_bianco',
            54: 'riso_sugo',
            55: 'roastbeef',
            56: 'rosbeef',
            57: 'rucola',
            58: 'salmone_-da_menu_sembra_spada_in_realta-',
            59: 'scaloppine',
            60: 'spinaci',
            61: 'stinco_di_maiale',
            62: 'strudel',
            63: 'torta_ananas',
            64: 'torta_cioccolato_e_pere',
            65: 'torta_crema',
            66: 'torta_crema_2',
            67: 'torta_salata_-alla_valdostana-',
            68: 'torta_salata_3',
            69: 'torta_salata_rustica_-zucchine-',
            70: 'torta_salata_spinaci_e_ricotta',
            71: 'yogurt',
            72: 'zucchine_impanate',
            73: 'zucchine_umido',
        }
        self.img_size = (1280, 1280)  # (640, 640)  # (height, width)

        # ---------------- model config ---------------- #

        scale = "yolox_x"  # "yolox_m" # "yolox_l" # "yolox_x"

        if scale == "yolox_s":
            self.depth = 0.33
            self.width = 0.50
        if scale == "yolox_m":
            self.depth = 0.67
            self.width = 0.75
        if scale == "yolox_l":
            self.depth = 1.0
            self.width = 1.0
        if scale == "yolox_x":
            self.depth = 1.33
            self.width = 1.25

        # self.ckpt = f"checkpoints/{scale}.pth"

    def merge(self, *args, **kwargs):
        super().merge(*args, **kwargs)
        check_exp_value(self)
