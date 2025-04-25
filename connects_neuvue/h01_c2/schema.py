import datajoint as dj
from .adapters import graph

h01 = dj.create_virtual_module('h01_process', 'h01_process')
schema = dj.Schema('h01_process')

@schema
class AutoProofreadNeuron(dj.Computed):
    definition='''
    -> h01.DecompositionCellType
    ---
    '''
    class Obj(dj.Part):
        definition="""
        ->master
        ---
        neuron_graph=NULL: <graph> #the graph for the 
        red_blue_suggestions=NULL: longblob
        """