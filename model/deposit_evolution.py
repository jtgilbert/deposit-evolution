import pandas as pd
import tqdm
import json
import logging
from dsp_transport.calculate_transport import transport
from typing import List

class DepositEvolution:
    def __init__(self, bound_conds: str, reach_atts: str, discharge: str, time_interval: int, width_hydro_geom: List,
                 depth_hydro_geom: List):

        self.boundary_conditions = json.loads(bound_conds)
        self.reaches = json.loads(reach_atts)
        self.w_geom = width_hydro_geom
        self.h_geom = depth_hydro_geom

        self.q = pd.read_csv(discharge)
        if 'Q' not in self.q.columns:
            raise Exception('Column "Q" not found in discharge csv')
        if 'Datetime' not in self.q.columns:
            raise Exception('Column "Datetime" not found in discharge csv')

        self.time_interval = time_interval

        # set up log file

        # set up output table
        self.out_table =

        # set initial time step and deposit volume in reaches dict
        self.reaches['timestep'] = self.q.loc[0, 'Datetime']

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

    def simulation(self):
        transfer_vals = {}

        for q in self.q.index:
            q_in = q * self.boundary_conditions['flow_factor']
            width_in = self.w_geom[0] * q_in ** self.w_geom[1]
            depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
            fractions_in = {float(s): f for s, f in self.boundary_conditions['gsd'].items()}
            qs_in_fractions = transport(fractions_in, self.boundary_conditions['slope'], q_in,
                                        depth_in, width_in, self.time_interval)
            transfer_vals["upstream"] = {"Qs_in": qs_in_fractions}

            for reach, attributes in self.reaches['reaches'].items():
                if reach == 'upstream':
                    qs_in = transfer_vals['upstream']['Qs_in']
                    q_in = q * attributes['flow_factor']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
                    fractions_in = {float(s): f for s, f in attributes['gsd'].items()}
                    slope = attributes['elevation'] - self.reaches['deposit_upstream']['elevation'] / attributes['length']
                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                             twod=True)










