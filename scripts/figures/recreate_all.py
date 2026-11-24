"""Recreate all ten figure pages from frozen records without fitting models."""
from pathlib import Path
import subprocess
import sys

def main():
    folder=Path(__file__).resolve().parent
    for script in ['round1_core.py','round2_design.py','round3_supplement.py']:
        subprocess.run([sys.executable,str(folder/script),'--draft'],check=True)
    subprocess.run([sys.executable,str(folder/'verify_numbers.py')],check=True)
    print('Recreated ten figure pages in generated/figures and generated/figure_previews. Numerical records checked; review font/layout differences before replacing publication exports.')

if __name__=='__main__':main()
