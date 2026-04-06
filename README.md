Might need to change the path in dataframes.py to specify where your .root files are. Right now it is just set to 'Data/'.

Must run the get_multiclass_bdt.py, get_wwdfdy_trained_bdt.py, and get_wwttbar_trained_bdt.py files before the get_efficiencies file.

Get_efficiencies should cleanly print out efficiencies of three different methods of analysis:
  1) All three processes (ww, dfdy, and ttbar) ran through a wwdfdy trained bdt. Then, events which have a bdt_score > 0.36 are kept and ran through a wwttbar trained bdt.
  2) Just ww and ttbar ran through a wwttbar trained bdt.
  3) All three processes ran through a multiclass bdt.

This script also creates a 'Plots' folder if there is not already one. If you want to change the name of the folder, there is a variable inside of the script that you just need to change in one place.
This script in particular just plots the bdt score distributions.
