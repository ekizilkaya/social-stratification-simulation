import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
from analysis import run_mobility_simulation, visualize_mobility_matrix

# Redirect stdout and stderr to prevent crash in windowed mode
class DummyWriter:
    def write(self, x): pass
    def flush(self): pass
sys.stdout = DummyWriter()
sys.stderr = DummyWriter()

def main():
    root = tk.Tk()
    root.withdraw()
    
    prompt = (
        "MACRO-SOCIOLOGICAL STRATIFICATION & MOBILITY SIMULATOR\n\n"
        "Select a macroeconomic structural scenario to simulate:\n"
        "  1. Rigid Imaginary (Hyper-stratified, dystopian)\n"
        "  2. United States Baseline (Moderate-high persistence)\n"
        "  3. Turkey Baseline (High persistence, strong elite closure)\n"
        "  4. Finland Baseline (Nordic model, high mobility)\n"
        "  5. Hyper-Mobile Imaginary (Perfect equality of opportunity)\n\n"
        "Enter the scenario number (1-5):"
    )
    
    choice = simpledialog.askstring("Scenario Selection", prompt, parent=root)
    
    if choice is None:
        sys.exit()
        
    scenarios = {
        "1": "Rigid Imaginary",
        "2": "United States",
        "3": "Turkey",
        "4": "Finland",
        "5": "Hyper-Mobile Imaginary"
    }
    
    scenario_name = scenarios.get(choice.strip())
    
    if not scenario_name:
        messagebox.showerror("Error", "Invalid selection. Terminating application.")
        sys.exit()
        
    messagebox.showinfo("Simulation Started", f"Compiling matrix for: {scenario_name}...\n\nExecuting agent accumulation cycles across 5 generations. Please wait, this may take a moment.")
    
    mobility_matrix = run_mobility_simulation(
        scenario=scenario_name, 
        generations=5, 
        num_agents=500, 
        width=20,
        height=20
    )
    
    messagebox.showinfo("Simulation Complete", "Simulation complete. Rendering visualizations.")
    visualize_mobility_matrix(mobility_matrix, scenario_name)

if __name__ == "__main__":
    main()