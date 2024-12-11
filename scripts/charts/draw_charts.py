import matplotlib.pyplot as plt

def draw_electricity_fli_results():
    fig = plt.figure()
    plt.title("Accuracy of NLinear model with SubsetRandomSampler compression")
    plt.ylabel('MSE')
    plt.xlabel('Compression ratio')

    train_len = 33505
    val_len = 10801
    total_len = train_len + val_len

    # Compressing train + val datasets
    tv_results = [
        {"ratio": 1, "mse": 0.07223959267139435, "mae":0.2063027024269104},
        {"ratio": 2, "mse":0.07332085818052292, "mae":0.21173346042633057},
        {"ratio": 4, "mse":0.07359476387500763, "mae":0.20944172143936157},
        {"ratio": 8, "mse":0.07715649902820587, "mae":0.21133197844028473},
        {"ratio": 16, "mse":0.08141212165355682, "mae":0.21738584339618683},
        {"ratio": 32, "mse":0.0848114863038063, "mae":0.2231469303369522},
        {"ratio": 64, "mse":0.09378495812416077, "mae":0.23618924617767334},
        {"ratio": 128, "mse":0.10068020224571228, "mae":0.24532315135002136},
        {"ratio": 256, "mse":0.10190605372190475, "mae":0.24772287905216217},
        {"ratio": 512, "mse":0.10524482280015945, "mae":0.2517310082912445},
        {"ratio": 1024, "mse":0.10695002228021622, "mae":0.25353124737739563},
        {"ratio": 2048, "mse":0.1063060536980629, "mae":0.25287675857543945},
        {"ratio": 4096, "mse":0.10931005328893661, "mae":0.25619950890541077},
        {"ratio": 8192, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 16384, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 32768, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 65536, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 131072, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 262144, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 524288, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 1048576, "mse":0.11079414933919907, "mae":0.2577025294303894},
    ]
    # ratios = [r["ratio"] for r in tv_results]
    # ratios = [total_len / (int(val_len/r["ratio"]) + int(train_len/r["ratio"])) for r in tv_results]
    ratios = []
    for res in tv_results:
        r = res["ratio"]
        compressed_len = int(val_len/r) + int(train_len/r)
        if compressed_len == 0:
            compressed_len = 1
        ratios.append(total_len / compressed_len)
    mses = [r["mse"] for r in tv_results]
    plt.plot(ratios, mses, label="Compressing train+val datasets")

    # Compressing train dataset only
    to_results = [
        {"ratio": 1, "mse":0.07231410592794418, "mae":0.2063920646905899},
        {"ratio": 2, "mse":0.07330863177776337, "mae":0.21168559789657593},
        {"ratio": 4, "mse":0.07342706620693207, "mae":0.21103587746620178},
        {"ratio": 8, "mse":0.0764801949262619, "mae":0.2100830227136612},
        {"ratio": 16, "mse":0.0821942389011383, "mae":0.2167663872241974},
        {"ratio": 32, "mse":0.0848114863038063, "mae":0.2231469303369522},
        {"ratio": 64, "mse":0.09378495812416077, "mae":0.23618924617767334},
        {"ratio": 128, "mse":0.10068020224571228, "mae":0.24532315135002136},
        {"ratio": 256, "mse":0.09982860833406448, "mae":0.24481399357318878},
        {"ratio": 512, "mse":0.10244469344615936, "mae":0.24891090393066406},
        {"ratio": 1024, "mse":0.1023879423737526, "mae":0.24857670068740845},
        {"ratio": 2048, "mse":0.10626213252544403, "mae":0.2528243064880371},
        {"ratio": 4096, "mse":0.10931005328893661, "mae":0.25619950890541077},
        {"ratio": 8192, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 16384, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 32768, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 65536, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 131072, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 262144, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 524288, "mse":0.11079414933919907, "mae":0.2577025294303894},
        {"ratio": 1048576, "mse":0.11079414933919907, "mae":0.2577025294303894},
    ]
    # to_ratios = [r["ratio"] for r in to_results]
    to_ratios = []
    for res in to_results:
        r = res["ratio"]
        compressed_len = val_len + int(train_len/r)
        if compressed_len == 0:
            compressed_len = 1
        to_ratios.append(total_len / compressed_len)
    to_mses = [r["mse"] for r in to_results]
    plt.plot(to_ratios, to_mses, label="Compressing training dataset only")

    # zoom a bit
    plt.axis([-500, 16384, 0.070, 0.113])

    plt.legend()
    fig.savefig("NLinear_accuracy_with_compressed_dataset.pdf", bbox_inches='tight')

def draw_RandomSampler_results():
	results = [
		{"ratio": 1, "mse":0.07218873500823975, "mae":0.20622242987155914},
        {"ratio": 2, "mse":0.07234717905521393, "mae":0.20638303458690643},
        {"ratio": 4, "mse":0.07273950427770615, "mae":0.20685532689094543},
        {"ratio": 8, "mse":0.07319291681051254, "mae":0.20738942921161652},
        {"ratio": 16, "mse":0.07442723214626312, "mae":0.20896165072917938},
        {"ratio": 32, "mse":0.07554087042808533, "mae":0.2105967402458191},
        {"ratio": 64, "mse":0.07755719125270844, "mae":0.2137523889541626},
        {"ratio": 128, "mse":0.07971423119306564, "mae":0.21733172237873077},
        {"ratio": 256, "mse":0.08276624977588654, "mae":0.2220194786787033},
        {"ratio": 512, "mse":0.09021437168121338, "mae":0.23237191140651703},
        {"ratio": 1024, "mse":0.0978069007396698, "mae":0.24210354685783386},
        {"ratio": 2048, "mse":0.1038285493850708, "mae":0.24949805438518524},
        {"ratio": 4096, "mse":0.10705295950174332, "mae":0.25335395336151123},
        {"ratio": 8192, "mse":0.11079414933919907, "mae":0.2577025294303894},
        # crashes after this (num_samples=0)
	]

	fig = plt.figure()
	plt.title("Accuracy of NLinear model with RandomSampler compression")
	plt.ylabel('MSE')
	plt.xlabel('Compression ratio')
    
	mses = [r["mse"] for r in results]
      
	train_len = 33505
	val_len = 10801
	total_len = train_len + val_len
     
	ratios = []
	for res in results:
		r = res["ratio"]
		compressed_len = int(val_len/r) + int(train_len/r)
		if compressed_len == 0:
			compressed_len = 1
		ratios.append(total_len / compressed_len)
    
	plt.plot(ratios, mses)
	fig.savefig("NLinear_accuracy_with_random_compression.pdf", bbox_inches='tight')

randomSampler_results = [
    {"preserved_points": 1, "mse":0.11079414933919907, "mae":0.2577025294303894},
    {"preserved_points": 2, "mse":0.11079414933919907, "mae":0.2577025294303894},
    {"preserved_points": 4, "mse":0.11079414933919907, "mae":0.2577025294303894},
    {"preserved_points": 8, "mse":0.10717432200908661, "mae":0.2534983158111572},
    {"preserved_points": 16, "mse":0.10404524952173233, "mae":0.24975897371768951},
    {"preserved_points": 32, "mse":0.09754296392202377, "mae":0.2417737990617752},
    {"preserved_points": 64, "mse":0.08986309915781021, "mae":0.23190705478191376},
    {"preserved_points": 128, "mse":0.0841890349984169, "mae":0.2240869104862213},
    {"preserved_points": 256, "mse":0.07971423119306564, "mae":0.21733172237873077},
    {"preserved_points": 512, "mse":0.07754780352115631, "mae":0.21375297009944916},
    {"preserved_points": 1024, "mse":0.0751136988401413, "mae":0.20998449623584747},
    {"preserved_points": 2048, "mse":0.07395604252815247, "mae":0.20830030739307404},
    {"preserved_points": 4096, "mse":0.07323602586984634, "mae":0.20742404460906982},
    {"preserved_points": 8192, "mse":0.07280801981687546, "mae":0.20694184303283691},
    {"preserved_points": 16384, "mse":0.07233629375696182, "mae":0.20639199018478394},
    {"preserved_points": 32768, "mse":0.0731145516037941, "mae":0.20789319276809692},
    {"preserved_points": 65536, "mse":0.07251517474651337, "mae":0.20666198432445526}
]

def draw_RandomSampler_inverted_results():
    mses = [r["mse"] for r in randomSampler_results]
      
    train_len = 33505
    val_len = 10801

    ratios = []
    for res in randomSampler_results:
        ratio = (train_len + val_len) / (2 * res["preserved_points"]) # keeping `preserved_points` points for both `train` and `val` datasets
        ratios.append(ratio)
    
    fig = plt.figure()
    plt.title("Accuracy of NLinear model with RandomSampler compression")
    plt.ylabel('MSE')
    plt.xlabel('Compression ratio')

    # zoom a bit
    plt.axis([-500, 9000, 0.070, 0.113])

    plt.plot(ratios, mses)
    fig.savefig("NLinear_accuracy_with_random_compression.pdf", bbox_inches='tight')

fli_results = [
    {"error": 0.001, "model_floats_count":1345665, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.002, "model_floats_count":1345665, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.004, "model_floats_count":1345665, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.008, "model_floats_count":1345662, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.016, "model_floats_count":1298886, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.032, "model_floats_count":1226643, "mse":0.07313929498195648, "mae":0.2069794237613678},
    {"error": 0.064, "model_floats_count":1073715, "mse":0.07313983887434006, "mae":0.2069801539182663},
    {"error": 0.128, "model_floats_count":864078, "mse":0.07314002513885498, "mae":0.20698142051696777},
    {"error": 0.256, "model_floats_count":629493, "mse":0.07314251363277435, "mae":0.20698542892932892},
    {"error": 0.512, "model_floats_count":402843, "mse":0.0731629952788353, "mae":0.20701350271701813},
    {"error": 1.024, "model_floats_count":204768, "mse":0.07324954122304916, "mae":0.2071533501148224},
    {"error": 2.048, "model_floats_count":80637, "mse":0.07356352359056473, "mae":0.20765310525894165},
    {"error": 4.096, "model_floats_count":29700, "mse":0.07961008697748184, "mae":0.2181011140346527},
    {"error": 8.192, "model_floats_count":10908, "mse":0.11287792772054672, "mae":0.2645367383956909},
    #{"error": 16.384, "model_floats_count":3618, "mse":1.0187914371490479, "mae":0.721802294254303},
    {"error": 32.768, "model_floats_count":273, "mse":0.3006591200828552, "mae":0.4017712473869324},
]

def draw_FLI_compression_results():
    rows_count = 69680
    cols_count = 7 + 1 # 7 data columns + date index
    raw_df_floats_count = rows_count * cols_count

    ratios = [raw_df_floats_count/r["model_floats_count"] for r in fli_results]
    mses = [r["mse"] for r in fli_results]

    fig = plt.figure()
    plt.title("Accuracy of NLinear model with FLI compression")
    plt.ylabel('MSE')
    plt.xlabel('Compression ratio')

    # zoom a bit
    #plt.axis([-500, 9000, 0.070, 0.113])

    plt.plot(ratios, mses, label="FLI constant compression")
    fig.savefig("NLinear_accuracy_with_FLI_compression.pdf", bbox_inches='tight')

def compare_compressions():
    fig = plt.figure()
    plt.title("Accuracy of NLinear model regarding train+val compression")
    plt.ylabel('MSE')
    plt.xlabel('Compression ratio')

    # original dataframe statistics
    rows_count = 69680
    cols_count = 7 + 1 # 7 data columns + date index
    raw_df_floats_count = rows_count * cols_count

    # random sampling compression
    rs_mses = [r["mse"] for r in randomSampler_results]
    rs_ratios = [raw_df_floats_count/(r["preserved_points"]*7) for r in randomSampler_results]
    plt.plot(rs_ratios, rs_mses, label="Random sampling")

    # fli compression
    fli_mses = [r["mse"] for r in fli_results]
    fli_ratios = [raw_df_floats_count/r["model_floats_count"] for r in fli_results]
    plt.plot(fli_ratios, fli_mses, label="FLI constant compression")

    # zoom a bit
    plt.axis([-500, 9000, 0.070, 0.113])

    plt.legend()
    fig.savefig("NLinear_compression.pdf", bbox_inches='tight')


# draw_electricity_fli_results()
# draw_RandomSampler_results()
# draw_RandomSampler_inverted_results()
# draw_FLI_compression_results()
compare_compressions()