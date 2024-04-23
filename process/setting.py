from process.produce_graph import obtain_N
class OPT():
    def __init__(self):
        self.lr = 1e-3
        self.weight_dacy = 5e-4  # for APS 1e-8
        self.batch_size = 32
        self.hidden_dim = 32
        self.num_layers = 2
        self.embed_dim = 32
        self.input_dim = 64
        self.d_ff = 128
        self.out_dim = 1
        self.time_embed_dim = 30
        self.N = obtain_N() + 1
        self.nhead = 4
        self.dropout = 0.1  # for APS 0.5
        self.d_v = 32
        self.max_grad_clip = 5  # for APS 10.0



