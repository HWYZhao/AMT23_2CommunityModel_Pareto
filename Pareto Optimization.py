#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution
from glob import glob
import re
import matplotlib.pyplot as plt

def objective_chla(params, X_chla, Y):
    P2, P3, P4, P5, _, _ = params
    P1 = 10**(0.08 * P2 + 0.66)
    
    MLD_pop = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2)))
    DCM_pop = P3*np.exp(-((Y - P4)/P5)**2.)
    model = MLD_pop + DCM_pop
    return np.sum((model - X_chla)**2)

def objective_cp(params, X_carbon, Y, S1):
    P2, P3, P4, P5, Cp1, Cp2 = params
    P1 = 10**(0.08 * P2 + 0.66)
    
    MLD_pop = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2)))
    DCM_pop = P3*np.exp(-((Y - P4)/P5)**2.)
    model = (Cp1*MLD_pop + Cp2*DCM_pop) * S1
    return np.sum((model - X_carbon)**2)

def validate_constraints(params, MLD_OD):
    P2, P3, P4, P5, Cp1, Cp2 = params
    P1 = 10**(0.08 * P2 + 0.66)
    
    if P1 <= 4.6 or P2 <= MLD_OD or P2 >= P4:
        return False
    if P4 <= MLD_OD or P4 <= 3 * P5:
        return False
    return True

def is_dominated(p1, p2):
    return (p2[0] <= p1[0] and p2[1] <= p1[1]) and (p2[0] < p1[0] or p2[1] < p1[1])

def select_best_solutions(population, pop_size):
    f1_vals = [p[1][0] for p in population]
    f2_vals = [p[1][1] for p in population]
    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)
    
    pareto = []
    for i, (x1, f1) in enumerate(population):
        dominated = False
        for j, (x2, f2) in enumerate(population):
            if i != j and is_dominated(
                ((f1[0] - f1_min)/(f1_max - f1_min), 
                 (f1[1] - f2_min)/(f2_max - f2_min)),
                ((f2[0] - f1_min)/(f1_max - f1_min), 
                 (f2[1] - f2_min)/(f2_max - f2_min))
            ):
                dominated = True
                break
        if not dominated:
            pareto.append((x1, f1))
    
    if len(pareto) < pop_size:
        remaining = sorted(
            [p for p in population if p not in pareto],
            key=lambda x: ((x[1][0] - f1_min)/(f1_max - f1_min))**2 + 
                         ((x[1][1] - f2_min)/(f2_max - f2_min))**2
        )
        pareto.extend(remaining[:pop_size-len(pareto)])
    
    return pareto[:pop_size]

def select_balanced_solution(population):
    f1_vals = [p[1][0] for p in population]
    f2_vals = [p[1][1] for p in population]
    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)
    
    best_score = float('inf')
    best_sol = None
    
    for x, (f1, f2) in population:
        f1_norm = (f1 - f1_min)/(f1_max - f1_min)
        f2_norm = (f2 - f2_min)/(f2_max - f2_min)
        score = np.sqrt(f1_norm**2 + f2_norm**2)
        
        if score < best_score:
            best_score = score
            best_sol = x
            
    return best_sol

def plot_results(population, selected_sol, f1_hist, f2_hist, param_hist, param_names, save_path=None, station_name=None):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    
    fig = plt.figure(figsize=(10, 8))
    
    # Layout params
    margins = {'top': 0.95, 'bottom': 0.05, 'left': 0.05, 'right': 0.95}
    col_gap = 0.10
    col_width = (margins['right'] - margins['left'] - col_gap) / 2
    
    left_col = margins['left']
    right_col = margins['left'] + col_width + col_gap
    
    # Heights
    total_height = margins['top'] - margins['bottom']
    param_h = (total_height - 5 * 0.015) / 6
    spacing = 0.015
    
    # Left plots
    left_h1 = param_h * 2
    left_h2 = param_h * 1.8
    left_h3 = param_h * 1.8
    
    top_bottom = margins['top'] - param_h
    left_y1 = top_bottom + param_h - left_h1
    left_y2 = left_y1 - left_h1 - 0.09
    left_y3 = left_y2 - left_h2 - 0.03
    
    # Pareto front
    ax1 = fig.add_axes([left_col, left_y1, col_width, left_h1])
    
    f1_vals = [p[1][0] for p in population]
    f2_vals = [p[1][1] for p in population]
    f1_min, f1_max = min(f1_vals), max(f1_vals)
    f2_min, f2_max = min(f2_vals), max(f2_vals)
    
    f1_norm = [(f - f1_min)/(f1_max - f1_min) for f in f1_vals]
    f2_norm = [(f - f2_min)/(f2_max - f2_min) for f in f2_vals]
    
    ax1.scatter(f1_norm, f2_norm, c='#DBA3B4', alpha=0.7, label='Population')
    
    if selected_sol:
        sel_f1 = (selected_sol[1][0] - f1_min)/(f1_max - f1_min)
        sel_f2 = (selected_sol[1][1] - f2_min)/(f2_max - f2_min)
        ax1.scatter(sel_f1, sel_f2, c='#2F2D54', s=80, label='Selected')
    
    ax1.set_xlabel(r'$\mathrm{RRS_{B,norm}}$')
    ax1.set_ylabel(r'$\mathrm{RRS_{Cp,norm}}$')
    ax1.set_title('Pareto Front')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(False)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    
    # Convergence plots
    ax2 = fig.add_axes([left_col, left_y2, col_width, left_h2])
    ax2.plot(f1_hist, color='#ACD1BE', linewidth=1.8)
    ax2.set_ylabel(r'$\mathrm{RRS_{B,2C}}$')
    ax2.set_xticklabels([])
    ax2.set_title('Convergence', pad=10)
    ax2.grid(False)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    ax3 = fig.add_axes([left_col, left_y3, col_width, left_h3])
    ax3.plot(f2_hist, color='#9192B3', linewidth=1.8)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel(r'$\mathrm{RRS_{Cp,2C}}$')
    ax3.grid(False)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))
    
    # Parameter plots
    colors = ['#547F9E', '#8B9FAA', '#B59478', '#C6B1A6', '#826AA2', '#B3AEBF']
    labels = [r'$\tau_1$', r'$B^*_{2,m}$', r'$\tau_2$', r'$\sigma$', r'$\theta_1$', r'$\theta_2$']
    
    for i, (param_vals, name, color) in enumerate(zip(param_hist, param_names, colors)):
        if not param_vals:
            continue
        
        y_pos = margins['top'] - param_h - i * (param_h + spacing)
        ax = fig.add_axes([right_col, y_pos, col_width, param_h])
        ax.plot(param_vals, color=color, linewidth=1.8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        
        if i == len(param_names) - 1:
            ax.set_xlabel('Generation')
        else:
            ax.set_xticklabels([])
            
        ax.set_ylabel(labels[i])
        
        if i == 0:
            ax.set_title("Convergence")
        ax.grid(False)
    
    plt.subplots_adjust(hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def pareto_optimization(bounds, args, MLD_OD, wuyi_params, pop_size=200, gens=100, plot_dir=None, station_name=None):
    X_chla, X_carbon, Y, S1 = args
    
    f1_hist = []
    f2_hist = []
    param_hist = [[] for _ in range(6)]
    
    # Init population
    population = []
    if validate_constraints(wuyi_params, MLD_OD):
        f1 = objective_chla(wuyi_params, X_chla, Y)
        f2 = objective_cp(wuyi_params, X_carbon, Y, S1)
        population.append((wuyi_params, (f1, f2)))
    
    attempts = 0
    while len(population) < pop_size and attempts < 1000:
        x = [np.random.uniform(b[0], b[1]) for b in bounds]
        if validate_constraints(x, MLD_OD):
            f1 = objective_chla(x, X_chla, Y)
            f2 = objective_cp(x, X_carbon, Y, S1)
            population.append((x, (f1, f2)))
        attempts += 1
    
    if not population:
        raise ValueError("Can't generate valid initial solutions")
    
    # Evolution
    for gen in range(gens):
        new_pop = []
        for i in range(pop_size):
            if len(population) >= 3:
                a, b, c = np.random.choice(len(population), 3, replace=False)
                x = list(population[a][0])
                
                for j in range(len(x)):
                    if np.random.random() < 0.7:
                        x[j] = x[j] + 0.7 * (population[b][0][j] - population[c][0][j])
                        x[j] = max(bounds[j][0], min(bounds[j][1], x[j]))
                
                if validate_constraints(x, MLD_OD):
                    f1 = objective_chla(x, X_chla, Y)
                    f2 = objective_cp(x, X_carbon, Y, S1)
                    new_pop.append((x, (f1, f2)))
        
        if new_pop:
            population.extend(new_pop)
            population = select_best_solutions(population, pop_size)
            
            best = select_balanced_solution(population)
            f1 = objective_chla(best, X_chla, Y)
            f2 = objective_cp(best, X_carbon, Y, S1)
            f1_hist.append(f1)
            f2_hist.append(f2)
            
            for i, param in enumerate(best):
                param_hist[i].append(param)
    
    # Save plot
    if plot_dir:
        final_dir = os.path.join(plot_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        
        prefix = f"{station_name}_" if station_name else ""
        selected = (select_balanced_solution(population), 
                   (objective_chla(select_balanced_solution(population), X_chla, Y),
                    objective_cp(select_balanced_solution(population), X_carbon, Y, S1)))
        
        plot_results(
            population, selected, f1_hist, f2_hist, param_hist,
            ['TAU1', 'BM2', 'TAU2', 'SIG2', 'Cp1', 'Cp2'],
            os.path.join(final_dir, f'{prefix}integrated_results.png'),
            station_name
        )
    
    return select_balanced_solution(population)

def process_file(file_path):
    try:
        filename = os.path.basename(file_path)
        station = os.path.splitext(filename)[0]
        
        skip_files = ['JR300_055.xlsx', 'JR300_060.xlsx', 'JR300_063.xlsx', 'JR300_065.xlsx']
        if filename in skip_files:
            print(f"Skip: {filename}")
            return False
            
        data = pd.read_excel(file_path)
        print(f"\nProcessing: {file_path}")
        
        # Add columns
        cols = ['P1_WUYI', 'TAU1_WUYI', 'BM2_WUYI', 'TAU2_WUYI', 'SIG2_WUYI', 
                'Cp1_WUYI', 'Cp2_WUYI', 'MLD_pop_WUYI', 'DCM_pop_WUYI',
                'model_CHL1_WUYI', 'model_CHL2_WUYI', 'model_TCHL_WUYI',
                'model_Cp1_WUYI', 'model_Cp2_WUYI', 'model_TCp_WUYI']
        for col in cols:
            data[col] = np.nan
        
        # Get params
        try:
            TAU1_init = data['TAU1_BOB_WUYI'].iloc[0]
            BM2_init = data['BM2_BOB_WUYI'].iloc[0]
            TAU2_init = data['TAU2_BOB_WUYI'].iloc[0]
            SIG2_init = data['SIG2_BOB_WUYI'].iloc[0]
            Cp1_init = data['Cp1_BOB_WUYI'].iloc[0]
            Cp2_init = data['Cp2_BOB_WUYI'].iloc[0]
            Scale_MLD_CHL = data['Scale_MLD_CHL'].iloc[0]
            MLD_OD = data['MLD_OD'].iloc[0]
        except Exception as e:
            print(f"Failed to get params: {str(e)}")
            return False
            
        # Extract data
        OPT_DIM = data['OPT_DIM'].to_numpy()
        CHL_DIM = data['CHL_DIM'].to_numpy()
        Beam_C = data['Smooth_Beam_C_corrected'].to_numpy()
        peak_idx = np.argmax(CHL_DIM)
        peak_depth = OPT_DIM[peak_idx]
        
        # Bounds
        bounds = [
            (MLD_OD, peak_depth),
            (0.0, 10.0),
            (MLD_OD, peak_depth*2),
            (0.1, TAU2_init),
            (0.001, 800),
            (0.001, 800)
        ]
        
        wuyi_params = [TAU1_init, BM2_init, TAU2_init, SIG2_init, Cp1_init, Cp2_init]
        args = (CHL_DIM, Beam_C, OPT_DIM, Scale_MLD_CHL)
        
        plot_dir = r"C:\Users\hz409\OneDrive - University of Exeter\Desktop\PhD_part2\Data\AMT23\Plots_WUYI2\321\pareto_plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        best = pareto_optimization(bounds, args, MLD_OD, wuyi_params, 
                                  plot_dir=plot_dir, station_name=station)
        
        # Unpack
        P2, P3, P4, P5, Cp1, Cp2 = best
        P1 = 10**(0.08 * P2 + 0.66)
        
        # Calculate
        MLD_pop = 1 - 1./(1+np.exp(-(P1/P2)*(OPT_DIM-P2)))
        DCM_pop = P3*np.exp(-((OPT_DIM - P4)/P5)**2.)
        
        # Save results
        data['P1_WUYI'] = P1
        data['TAU1_WUYI'] = P2
        data['BM2_WUYI'] = P3
        data['TAU2_WUYI'] = P4
        data['SIG2_WUYI'] = P5
        data['Cp1_WUYI'] = Cp1
        data['Cp2_WUYI'] = Cp2
        data['MLD_pop_WUYI'] = MLD_pop
        data['DCM_pop_WUYI'] = DCM_pop
        
        data['model_CHL1_WUYI'] = MLD_pop * Scale_MLD_CHL
        data['model_CHL2_WUYI'] = DCM_pop * Scale_MLD_CHL
        data['model_TCHL_WUYI'] = data['model_CHL1_WUYI'] + data['model_CHL2_WUYI']
        
        data['model_Cp1_WUYI'] = Cp1 * MLD_pop * Scale_MLD_CHL
        data['model_Cp2_WUYI'] = Cp2 * DCM_pop * Scale_MLD_CHL
        data['model_TCp_WUYI'] = data['model_Cp1_WUYI'] + data['model_Cp2_WUYI']
        
        data.to_excel(file_path, index=False)
        print(f"Done: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def generate_summary(base_path):
    results = []
    params = ['P1', 'TAU1', 'BM2', 'TAU2', 'SIG2', 'Cp1', 'Cp2']
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            files = glob(os.path.join(folder_path, 'JR300_*.xlsx'))
            files = [f for f in files if re.match(r'JR300_\d+\.xlsx$', os.path.basename(f))]
            
            for file in files:
                try:
                    data = pd.read_excel(file)
                    station = os.path.basename(file).replace('.xlsx', '')
                    
                    # Get initial values
                    init_params = {
                        'P1': 10**(0.08 * data['TAU1_BOB_WUYI'].iloc[0] + 0.66),
                        'TAU1': data['TAU1_BOB_WUYI'].iloc[0],
                        'BM2': data['BM2_BOB_WUYI'].iloc[0],
                        'TAU2': data['TAU2_BOB_WUYI'].iloc[0],
                        'SIG2': data['SIG2_BOB_WUYI'].iloc[0],
                        'Cp1': data['Cp1_BOB_WUYI'].iloc[0],
                        'Cp2': data['Cp2_BOB_WUYI'].iloc[0]
                    }
                    
                    # Get optimized values
                    opt_params = {
                        'P1': data['P1_WUYI'].iloc[0],
                        'TAU1': data['TAU1_WUYI'].iloc[0],
                        'BM2': data['BM2_WUYI'].iloc[0],
                        'TAU2': data['TAU2_WUYI'].iloc[0],
                        'SIG2': data['SIG2_WUYI'].iloc[0],
                        'Cp1': data['Cp1_WUYI'].iloc[0],
                        'Cp2': data['Cp2_WUYI'].iloc[0]
                    }
                    
                    # Calculate changes
                    for param in params:
                        if pd.notna(init_params[param]) and pd.notna(opt_params[param]):
                            if init_params[param] != 0:
                                change = (opt_params[param] - init_params[param]) / init_params[param] * 100
                                formatted = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                            else:
                                formatted = "N/A"
                        else:
                            formatted = "N/A"
                        
                        results.append({
                            'Station': station,
                            'Parameter': param,
                            'Initial': init_params[param],
                            'Optimized': opt_params[param],
                            'Change': formatted
                        })
                
                except Exception as e:
                    print(f"Error in {file}: {str(e)}")
    
    if results:
        df = pd.DataFrame(results)
        pivot = df.pivot(index='Station', columns='Parameter', values='Change')
        pivot = pivot[params]
        
        df.to_excel(os.path.join(base_path, 'Parameter_Change_Detailed.xlsx'), index=False)
        pivot.to_excel(os.path.join(base_path, 'Parameter_Change_Summary.xlsx'))
        
        print("Summary saved")

def main():
    base_path = r"C:\Users\hz409\OneDrive - University of Exeter\Desktop\PhD_part2\Data\AMT23"
    print("Starting...")
    
    count = 0
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            files = glob(os.path.join(folder_path, 'JR300_*.xlsx'))
            files = [f for f in files if re.match(r'JR300_\d+\.xlsx$', os.path.basename(f))]
            
            for file in files:
                print(f'\nProcessing {os.path.basename(file)} in {folder}')
                if process_file(file):
                    count += 1
    
    print(f"\nDone! Processed {count} files.")
    
    if count > 0:
        print("\nGenerating summary...")
        generate_summary(base_path)

if __name__ == "__main__":
    main()

