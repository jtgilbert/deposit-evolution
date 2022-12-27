import pandas as pd
import tqdm
import json
import logging
from dsp_transport.calculate_transport import transport
from typing import Dict

class DepositEvolution:
    def __init__(self, bound_conds: Dict, reach_atts: Dict, discharge: str, time_interval: int):

        self.boundary_conditions = bound_conds
        self.reaches = reach_atts

        self.q = pd.read_csv(discharge)
        if 'Q' not in self.q.columns:
            raise Exception('Column "Q" not found in discharge csv')

        self.time_interval = time_interval

        # set up log file

        # set up output table
        self.out_table = 

    def exner(self, d_qs_bed, reach_length, porosity):

        dz_dt = (1 / (1-porosity)) * (d_qs_bed / reach_length)
        dz = dz_dt * self.time_interval

        return dz

    def delta_width(self, d_qs_wall, reach_length, porosity):

        dw_dt = (1 / (1-porosity)) * (d_qs_wall / reach_length)
        dw = dw_dt * self.time_interval

        return dw

    def serialize_timestep(self, outfile):

        with open(outfile, 'w') as dst:
            json.dump(dst, self.out_table, indent=4)








