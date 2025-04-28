import datajoint as dj
from ..utils import (
    file_utils as fileu,
    nx_graph_utils as nxgu,
    dj_utils as dju,
    split_suggestion_utils as ssu,
)


class API:
    def __init__(
        self,
        secret_password = None,
        secret_dict=None,
        **kwargs):
        
        if secret_dict is None:
            secret_dict = self.secret_dict_from_password(secret_password)
        
        dj.config['database.password'] = secret_dict['password']
        dj.config['database.username'] = secret_dict['username']
        dj.conn()
        
        from .schema import AutoProofreadNeuron
        
        self.autoproof_table = AutoProofreadNeuron
        self.autoproof_obj_table = self.autoproof_table.Obj
        
    def secret_dict_from_password(password,username='admin'):
        return {'username': username, 'password': password}

    def nx_graph_autoproof_from_segment_id(
        self,
        segment_id,
        split_index = 0,
        add_bounding_box = True):
        restr_table = (self.autoproof_obj_table & dict(
            segment_id=segment_id,
            split_index = split_index
            )
        )
        G_file = restr_table.fetch1('neuron_graph')
        G = fileu.decompress_pickle(G_file)
        if add_bounding_box:
            nxgu.add_bounding_boxes_to_graph(G)
        return G
    
    def autoproof_segment_metadata_df(self,table=None):
        if table is None:
            table = self.autoproof_table
        return dju.df_from_table(table)
    
    
    def red_blue_split_df_from_segment_id(self,segment_id,split_index = 0,return_df = True):
        
        key = dict(segment_id=segment_id,split_index = split_index)
        red_blue_suggestions = (self.autoproof_obj_table & key).fetch1("red_blue_suggestions")
        if red_blue_suggestions is None:
            print(f"the red_blue_suggestions saved were None")
            return None
        return ssu.red_blue_suggestion_dicts(red_blue_suggestions,return_df=return_df)
    