import pandas as pd
import os
from tqdm import tqdm
import json
import logging
from dsp_transport.calculate_transport import transport
from typing import List


class DepositEvolution:
    def __init__(self, bound_conds: str, reach_atts: str, discharge: str, time_interval: int, width_hydro_geom: List,
                 depth_hydro_geom: List, reach_name: str):

        f_bc = open(bound_conds)
        self.boundary_conditions = json.load(f_bc)
        f_ra = open(reach_atts)
        self.reaches = json.load(f_ra)
        self.w_geom = width_hydro_geom
        self.h_geom = depth_hydro_geom
        self.reach_name = reach_name

        self.q = pd.read_csv(discharge)
        if 'Q' not in self.q.columns:
            raise Exception('Column "Q" not found in discharge csv')
        if 'Datetime' not in self.q.columns:
            raise Exception('Column "Datetime" not found in discharge csv')

        self.time_interval = time_interval

        # set up log file
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Outputs')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'Outputs'))
        logfile = f'../Outputs/{self.reach_name}.log'
        logging.basicConfig(filename=logfile, format='%(levelname)s:%(message)s', level=logging.DEBUG)
        logging.info('Initializing model setup')

        # set up output table
        # self.out_table =

        # set initial time step and deposit volume in reaches dict
        self.reaches['timestep'] = self.q.loc[0, 'Datetime']

    def exner(self, d_qs_bed, reach_length, reach_width, porosity):

        volume = d_qs_bed / ((1-porosity)*2650)
        dz = volume / reach_length * reach_width

        return dz

    def delta_width(self, d_qs_wall, reach_length, depth, porosity):

        volume = d_qs_wall / ((1 - porosity)*2650)
        dw = volume / ((reach_length * depth) * 2)

        return dw

    def percentiles(self, fractions):
        out_sizes = []
        d_percentiles = [0.5, 0.84]
        for p in d_percentiles:
            cumulative = 0
            grain_size = None
            # while cumulative < p:
            for size, fraction in fractions.items():
                if cumulative < p:
                    cumulative += fraction
                    grain_size = float(size)
            for i, size in enumerate(fractions.keys()):
                if float(size) == grain_size:
                    low_size = float(list(fractions.keys())[i - 1]) - 0.25
                    low_cum = cumulative - float(fractions[list(fractions.keys())[i - 1]])
                    high_size = grain_size - 0.25
                    high_cum = cumulative
                    p_out = ((low_cum / p) * low_size + (p / high_cum) * high_size) / ((low_cum / p) + (p / high_cum))
                    out_sizes.append(p_out)

        return out_sizes[0], out_sizes[1]

    def serialize_timestep(self, outfile):

        with open(outfile, 'w') as dst:
            json.dump(dst, self.reaches, indent=4)

    def sediment_calculations(self, qsinputs: dict, qsoutputs: dict):

        if len(qsinputs.keys()) != 2:
            tot_in = sum([rate[1] for size, rate in qsinputs.items()])
        else:
            tot_in = sum([rate[1] for size, rate in qsinputs['bed'].items()]) + \
                     sum([rate[1] for size, rate in qsinputs['wall'].items()])

        if len(qsoutputs.keys()) != 2:
            tot_out = sum([rate[1] for size, rate in qsoutputs.items()])
            d_qs_bed = tot_in - tot_out
            d_qs_wall = None

        else:
            tot_bed = sum([rate[1] for size, rate in qsoutputs['bed'].items()])
            d_qs_wall = sum([rate[1] for size, rate in qsoutputs['wall'].items()])
            tot_out = tot_bed + d_qs_wall
            d_qs_bed = tot_in - tot_bed

        return d_qs_bed, d_qs_wall

    def simulation(self):
        transfer_vals = {
            "upstream": {"Qs_out": None},
            "deposit_upstream": {"Qs_in": None, "Qs_out": None},
            "deposit_downstream": {"Qs_in": None, "Qs_out": None},
            "downstream": {"Qs_in": None, "Qs_out": None}
        }

        for i in tqdm(self.q.index):
            q_in = self.q.loc[i, 'Q'] * self.boundary_conditions['flow_scale']
            width_in = self.w_geom[0] * q_in ** self.w_geom[1]
            depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
            fractions_in = {float(s): f for s, f in self.boundary_conditions['upstream_gsd'].items()}
            qs_fractions = transport(fractions_in, self.boundary_conditions['upstream_slope'], q_in,
                                        depth_in, width_in, self.time_interval)
            transfer_vals['upstream']['Qs_in'] = qs_fractions

            for reach, attributes in self.reaches['reaches'].items():
                if reach == 'upstream':
                    qs_in = transfer_vals['upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
                    fractions_in = {float(s): f for s, f in attributes['gsd'].items()}
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_upstream']['elevation']) / attributes['length']
                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval)
                    transfer_vals['upstream']['Qs_out'] = qs_fractions
                    transfer_vals['deposit_upstream']['Qs_in'] = qs_fractions

                    d_qs_bed, d_qs_wall = self.sediment_calculations(transfer_vals['upstream']['Qs_in'],
                                                                     transfer_vals['upstream']['Qs_out'])
                    dz = self.exner(d_qs_bed, attributes['length'], attributes['bankfull_width'], 0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['upstream']['elevation'] = attributes['elevation'] + dz

                    # update fractions
                    d50, d84 = self.percentiles(attributes['gsd'])
                    d50, d84 = (2**-d50)/1000, (2**-d84)/1000
                    volume = max((d84 / 2), d50) * attributes['length'] * attributes['bankfull_width']
                    for size, fraction in attributes['gsd'].items():
                        change_mass = qs_in[float(size)][1] - qs_fractions[float(size)][1]
                        change_vol = change_mass / 2093
                        existing_vol = fraction * volume
                        new_vol = existing_vol + change_vol
                        self.reaches['reaches']['upstream']['gsd'][size] = new_vol / volume

                if reach == 'deposit_upstream':
                    qs_in = transfer_vals['deposit_upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = attributes['width']
                    depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
                    fractions_in = {float(s): f for s, f in attributes['gsd_bed'].items()}
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_downstream']['elevation']) / \
                            attributes['length']
                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                             twod=True)
                    transfer_vals['deposit_upstream']['Qs_out'] = qs_fractions

                    d_qs_bed, d_qs_wall = self.sediment_calculations(transfer_vals['deposit_upstream']['Qs_in'],
                                                                     transfer_vals['deposit_upstream']['Qs_out'])
                    dz = self.exner(d_qs_bed, attributes['length'], attributes['width'],
                                    0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['deposit_upstream']['elevation'] = attributes['elevation'] + dz
                    dw = self.delta_width(d_qs_wall, attributes['length'], attributes['width'], 0.21)
                    self.reaches['reaches']['deposit_upstream']['width'] = attributes['width'] + dw

                    # update fractions
                    d50_b, d84_b = self.percentiles(attributes['gsd_bed'])
                    d50_bed, d84_bed = (2 ** -d50_b)/1000, (2 ** -d84_b)/1000
                    volume_bed = max((d84_bed / 2), d50_bed) * attributes['length'] * max(attributes['bankfull_width'],
                                                                                          attributes['width'])
                    for size, fraction in attributes['gsd_bed'].items():
                        change_mass = qs_in[float(size)][1] - qs_fractions['bed'][float(size)][1]
                        change_vol = change_mass / 2093
                        existing_vol = fraction * volume_bed
                        new_vol = existing_vol + change_vol
                        self.reaches['reaches']['deposit_upstream']['gsd_bed'][size] = new_vol / volume_bed

                    d50_w, d84_w = self.percentiles(attributes['gsd_wall'])
                    d50_wall, d84_wall = (2 ** -d50_w)/1000, (2 ** - d84_w)/1000
                    volume_wall = (max((d84_wall / 2), d50_wall) * attributes['length'] * depth_in) * 2
                    for size, fraction in attributes['gsd_wall'].items():
                        change_mass = -qs_fractions['wall'][float(size)][1]
                        change_vol = change_mass / 2093  # the change from transport from walls
                        if dz < 0:
                            add_volume = (max((d84_wall / 2), d50_wall) * attributes['length'] * -dz) * 2
                            frac_volume = self.boundary_conditions['deposit_gsd'][size] * add_volume
                            change_vol = change_vol + frac_volume
                        else:
                            add_volume = 0
                        existing_vol = fraction * volume_wall
                        new_vol = existing_vol + change_vol

                        self.reaches['reaches']['deposit_upstream']['gsd_wall'][size] = new_vol / (volume_wall + add_volume)




b_c = '../Inputs/boundary_conditions.json'
r_a = '../Inputs/reaches.json'
dis = '../Inputs/Woods_Q.csv'
t_i = 900
whg = [5.947, 0.115]
dhg = [0.283, 0.402]
r_n = 'Woods'

inst = DepositEvolution(b_c, r_a, dis, t_i, whg, dhg, r_n)
inst.simulation()
