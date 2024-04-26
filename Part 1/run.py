from AutoSys2D import FitzHugh_Nagumo


FHN_std_parameters = {'a': 0.7, 
                  'b': 0.8,
                  'tau': 12.5,
                  'r': 0.1,
                  'i_ext': 3.2418,
}
FHN_cell = FitzHugh_Nagumo(x0=-1.5, y0=-1., parameters=FHN_std_parameters, run_time=500, front_end=True)
#FHN_cell.run()
FHN_cell.solve()
FHN_cell.animation()