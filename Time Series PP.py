import os
import tempfile
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.timeseries import DFData, OutputWriter, run_timeseries
from pandapower.control import ConstControl
from pandapower.control import ConstControl, DiscreteTapControl
from pandapower.run import set_user_pf_options
import matplotlib.pyplot as plt

rng = np.random.default_rng(10)

# Creazione di una rete di test con 10 bus, 2 linee, 2 trasformatori, 6 carichi e 7 generatori statici, i generatori controllabili non lo sono veramente, 
# perche constcontrol sovrappone i profili di generazione  
def simple_test_net():
    net = pp.create_empty_network()
    set_user_pf_options(net, init_vm_pu="flat", init_va_degree="dc", calculate_voltage_angles=True)

    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    b3 = pp.create_bus(net, 20)
    b4 = pp.create_bus(net, 20)
    b5 = pp.create_bus(net, 20)
    b6 = pp.create_bus(net, 0.4)
    b7 = pp.create_bus(net, 20)
    b8_mt = pp.create_bus(net, 20)
    b8 = pp.create_bus(net, 0.4)
    b9 = pp.create_bus(net, 0.4)

    pp.create_ext_grid(net, b0)
    pp.create_line(net, b0, b1, 10, "48-AL1/8-ST1A 110.0")
    pp.create_transformer(net, b1, b2, "25 MVA 110/20 kV")
    pp.create_line(net, b2, b4, 2, "305-AL1/39-ST1A 110.0")
    
    pp.create_line(net, b4, b8_mt, 2.0, "184-AL1/30-ST1A 20.0")
    pp.create_transformer(net, b8_mt, b8, "0.25 MVA 20/0.4 kV")

    pp.create_transformer(net, b4, b6, "0.25 MVA 20/0.4 kV")
    pp.create_transformer(net, b1, b3, "25 MVA 110/20 kV")
    pp.create_line(net, b3, b5, 2, "243-AL1/39-ST1A 20.0")
    pp.create_line(net, b5, b7, 2, "184-AL1/30-ST1A 20.0")
    pp.create_transformer(net, b7, b9, "0.25 MVA 20/0.4 kV")

    pp.create_load(net, b2, p_mw=12, q_mvar=3, name='load1')
    pp.create_load(net, b3, p_mw=15, q_mvar=4, name='load2')
    pp.create_load(net, b5, p_mw=10, q_mvar=2, name='load3')
    pp.create_load(net, b6, p_mw=0.1, q_mvar=0.02, name='load4')
    pp.create_load(net, b8, p_mw=0.3, q_mvar=0.03, name='load5')
    pp.create_load(net, b9, p_mw=0.1, q_mvar=0.02, name='load6')
    pp.create_sgen(net, b4, p_mw=20, q_mvar=0.15, name='sgen1', controllable=True, min_p_mw=0.2, max_p_mw=20)
    pp.create_sgen(net, b5, p_mw=15, q_mvar=0.2, name='sgen2', controllable=True, min_p_mw=1, max_p_mw=15)
    pp.create_sgen(net, b6, p_mw=0.3, q_mvar=0.01, name='sgen3', controllable=True, min_p_mw=0.01, max_p_mw=0.3)
    pp.create_sgen(net, b7, p_mw=13, q_mvar=2, name='sgen4', controllable=False)
    pp.create_sgen(net, b8, p_mw=0.09, q_mvar=0.01, name='sgen5', controllable=False)
    pp.create_sgen(net, b8, p_mw=0.15, q_mvar=0.02, name='sgen6', controllable=False)
    pp.create_sgen(net, b9, p_mw=0.1, q_mvar=0.02, name='sgen7', controllable=False)
    

    return net

 # Creazione dei profili di carico e generazione con andamento temporale realistico (non casuale) per 100 step temporali
def create_data_source(n_timesteps, net):
    load_P_rng = list()
    gen_P_rng = list()
    load_Q_rng = list()
    gen_Q_rng = list()
    for i in range(6):

        v = rng.normal(loc=0.9, scale=0.1, size=n_timesteps)
        cut = len(v) // 2

        v[:cut] = sorted(v[:cut])
        v[cut:] = sorted(v[cut:], reverse=True)

        load_P_rng.append(v * net.load.p_mw.values[i])

    for i in range(6):
        
        v = rng.normal(loc=0.8, scale=0.1, size=n_timesteps)
        cut = len(v) // 2

        v[:cut] = sorted(v[:cut])
        v[cut:] = sorted(v[cut:], reverse=True)

        load_Q_rng.append(v * net.load.q_mvar.values[i])

    for i in range(7):
        
        v = rng.lognormal(mean=0, sigma=0.1, size=n_timesteps)
        cut = len(v) // 2

        v[:cut] = sorted(v[:cut])
        v[cut:] = sorted(v[cut:], reverse=True)

        gen_P_rng.append(v * net.sgen.p_mw.values[i])
    for i in range(7):
                
        v = rng.lognormal(mean=0, sigma=0.1, size=n_timesteps)
        cut = len(v) // 2

        v[:cut] = sorted(v[:cut])
        v[cut:] = sorted(v[cut:], reverse=True)

        gen_Q_rng.append(v * net.sgen.q_mvar.values[i])  



    # Creazione dizionario per il DataFrame
    profiles_dict = {}
    for i in range(6):
        profiles_dict[f"load{i+1}_p"] = load_P_rng[i]
        profiles_dict[f"load{i+1}_q"] = load_Q_rng[i]
    for i in range(7):
        profiles_dict[f"sgen{i+1}_p"] = gen_P_rng[i]
        profiles_dict[f"sgen{i+1}_q"] = gen_Q_rng[i]

    profiles = pd.DataFrame(profiles_dict)

    # stampa i primi valori per verifica
    print("Profili caricati correttamente. Esempio load1_p:")
    print(profiles["load1_p"].head())

    return DFData(profiles)


def create_controllers(net, ds):
    # Gestione automatizzata dei carichi (load 1-6)
    for i in range(1, 7):
        load_name = f"load{i}"
        idx = net.load.index[net.load.name == load_name]
        ConstControl(net, "load", "p_mw", element_index=idx, data_source=ds, profile_name=f"{load_name}_p")
        ConstControl(net, "load", "q_mvar", element_index=idx, data_source=ds, profile_name=f"{load_name}_q")

    # Gestione automatizzata dei generatori statici (gen 1-7)
    for i in range(1, 8):
        sgen_name = f"sgen{i}"
        idx = net.sgen.index[net.sgen.name == sgen_name]
        ConstControl(net, "sgen", "p_mw", element_index=idx, data_source=ds, profile_name=f"{sgen_name}_p")
        ConstControl(net, "sgen", "q_mvar", element_index=idx, data_source=ds, profile_name=f"{sgen_name}_q")
    
    # Controllo tap changer per il trasformatore
    # Regola la tensione al secondario del primo trasformatore (indice 0) per mantenerla tra 0.98 e 1.02 pu
    DiscreteTapControl(net, tid=0, vm_upper_pu=1.02, vm_lower_pu=0.98, side="lv")
    

# Configurazione dell'OutputWriter per salvare i risultati in file Excel
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')
    ow.log_variable('res_load', 'q_mvar')     

    
    return ow

# Esecuzione della simulazione time series con gestione automatizzata dei carichi e generatori, e salvataggio dei risultati in file Excel
def timeseries_example():
    net = simple_test_net()
    n_timesteps = 100
    time_steps = range(n_timesteps)

    ds = create_data_source(n_timesteps, net)
    create_controllers(net, ds)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(script_dir, "time_series_example")
    os.makedirs(output_dir, exist_ok=True)

    create_output_writer(net, time_steps, output_dir)
    run_timeseries(net, time_steps, continue_on_divergence=True)

    print(net.res_line.loading_percent)
    print("Simulazione completata. Grafico in corso...")
    
    # Caricamento e visualizzazione dei risultati della tensione dai file generati
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    if os.path.exists(vm_pu_file):
        vm_res = pd.read_excel(vm_pu_file, index_col=0)
        vm_res.plot(title="Andamento Tensione Bus (pu)", figsize=(10, 5))
        plt.xlabel("Step Temporale")
        plt.ylabel("Tensione [pu]")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


timeseries_example()
