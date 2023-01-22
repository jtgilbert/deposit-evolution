import pandas as pd
import numpy as np
from math import atan
import os
from tqdm import tqdm
import json
import logging
from dsp_transport.calculate_transport import transport
from typing import List


def exner(d_qs_bed, reach_length, reach_width, porosity):
    volume = d_qs_bed / ((1 - porosity) * 2650)
    dz = volume / (reach_length * reach_width)

    return dz


def delta_width(d_qs_wall, reach_length, depth, porosity):

    volume = d_qs_wall / ((1 - porosity) * 2650)
    dw = volume / ((reach_length * depth) * 2)

    return dw


def percentiles(fractions):
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


def sediment_calculations(qsinputs: dict, qsoutputs: dict):

    if len(qsinputs.keys()) != 2:
        in_vals = [rate[1] for size, rate in qsinputs.items()]
        tot_in = sum(in_vals)
    else:
        in_vals = [[rate[1] for size, rate in qsinputs['bed'].items()],
                   [rate[1] for size, rate in qsinputs['wall'].items()]]
        tot_in = sum(in_vals[0]) + sum(in_vals[1])

    if len(qsoutputs.keys()) != 2:
        out_vals = [rate[1] for size, rate in qsoutputs.items()]
        tot_out = sum(out_vals)
        d_qs_bed = tot_in - tot_out
        d_qs_wall = None

    else:
        bed_vals = [rate[1] for size, rate in qsoutputs['bed'].items()]
        tot_bed = sum(bed_vals)
        wall_vals = [rate[1] for size, rate in qsoutputs['wall'].items()]
        d_qs_wall = sum(wall_vals)
        tot_out = tot_bed + d_qs_wall
        d_qs_bed = tot_in - tot_bed

    return d_qs_bed, d_qs_wall


def update_fractions(fractions_in, d_qs_mass, length, width, thickness=None, depth=None, fractype=None):
    if fractype is None:
        ex_volume = length * width * thickness
        frac_volumes = [frac * ex_volume for size, frac in fractions_in.items()]
        change_volumes = [sedyield / 2093 for size, sedyield in d_qs_mass.items()]
    elif fractype == 'bed':
        ex_volume = length * width * thickness
        frac_volumes = [frac * ex_volume for size, frac in fractions_in.items()]
        change_volumes = [sedyield/2093 for size, sedyield in d_qs_mass['bed'].items()]
    elif fractype == 'wall':
        ex_volume = length * depth*2 * width
        frac_volumes = [frac * ex_volume for size, frac in fractions_in.items()]
        change_volumes = [sedyield / 2093 for size, sedyield in d_qs_mass['wall'].items()]
    else:
        raise Exception('fractype must be "bed" or "wall"')
    dif = [frac + change_volumes[i] for i, frac in enumerate(frac_volumes)]
    new_volume = ex_volume + sum(change_volumes)
    fractions_out = {key: dif[i] / new_volume for i, key in enumerate(list(fractions_in.keys()))}
    total = sum([frac for size, frac in fractions_out.items()])

    if total > 1.1:
        raise Exception('Fractions sum to more than 1')

    return fractions_out


def update_fractions_in(fractions_in, qs_in, active_volume, porosity):
    if len(qs_in.keys()) == 2:
        qs_mass = [qs_in['bed'][size][1] + qs_in['wall'][size][1] for size, value in qs_in['bed'].items()]
    else:
        qs_mass = [value[1] for size, value in qs_in.items()]

    qs_vol = [mass / ((1-porosity)*2650) for mass in qs_mass]
    existing_vols = [frac * active_volume for size, frac in fractions_in.items()]
    new_vols = [vol + existing_vols[i] for i, vol in enumerate(qs_vol)]

    fractions_out = {key: new_vols[i] / sum(new_vols) for i, key in enumerate(list(fractions_in.keys()))}

    total = sum([frac for size, frac in fractions_out.items()])
    if total > 1.1:
        raise Exception('Fractions sum to more than 1')

    return fractions_out


def update_fractions_out(fractions, qs_out, active_volume, porosity):
    # run separate for bed and wall fractions
    qs_mass = [value[1] for size, value in qs_out.items()]

    qs_vol = [-mass / ((1-porosity)*2650) for mass in qs_mass]
    existing_vols = [frac * active_volume for size, frac in fractions.items()]
    new_vols = [vol + existing_vols[i] for i, vol in enumerate(qs_vol)]

    fractions_out = {key: new_vols[i] / sum(new_vols) for i, key in enumerate(list(fractions.keys()))}

    total = sum([frac for size, frac in fractions_out.items()])
    if total > 1.1:
        raise Exception('Fractions sum to more than 1')

    return fractions_out


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
        reach = [key for key in self.reaches['reaches'].keys()]
        timestep = self.q['Datetime']
        iterables = [reach, timestep]
        index = pd.MultiIndex.from_product(iterables, names=['Reach', 'DateTime'])
        zeros = np.zeros((len(reach) * len(timestep), 4))
        self.out_df = pd.DataFrame(zeros, index=index, columns=['elev', 'width', 'D50', 'yield'])

        self.init_elevs = {key: self.reaches['reaches'][key]['elevation'] for key in self.reaches['reaches'].keys()}

    def serialize_timestep(self, outfile):

        with open(outfile, 'w') as dst:
            json.dump(self.reaches, dst, indent=4)

    def save_df(self):
        logging.info('Saving output csv')
        self.out_df.to_csv(f'../Outputs/{self.reach_name}_out.csv')

    def incision_feedback(self, reach, incision_depth):
        d50_bed, d84_bed = percentiles(self.reaches['reaches'][reach]['gsd_bed'])
        d50_bed, d84_bed = 2**-d50_bed / 1000, 2**-d84_bed / 1000
        d50_wall, d84_wall = percentiles(self.reaches['reaches'][reach]['gsd_wall'])
        d50_wall, d84_wall = 2**-d50_wall / 1000, 2**-d84_wall / 1000

        bed_vol = d50_bed * self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['bankfull_width']
        wall_vol = d50_wall * self.reaches['reaches'][reach]['length'] * incision_depth
        tot_vol = bed_vol + wall_vol

        # new bed gsd
        self.reaches['reaches'][reach]['gsd_bed'] = \
            {key: ((self.reaches['reaches'][reach]['gsd_bed'][key]*bed_vol) +
                  (self.reaches['reaches'][reach]['gsd_wall'][key]*wall_vol)) /
                  tot_vol for key in self.reaches['reaches'][reach]['gsd_bed'].keys()}

        # new wall gsd
        bound_gsd = {float(s): frac for s, frac in self.boundary_conditions['deposit_gsd'].items()}
        self.reaches['reaches'][reach]['gsd_wall'] = \
            {key: ((self.reaches['reaches'][reach]['gsd_wall'][key] * (0.5*wall_vol)) +
                  (bound_gsd[key] * (0.5*wall_vol))) /
                  wall_vol for key in self.reaches['reaches'][reach]['gsd_wall'].keys()}

        # adjust width
        dw = wall_vol / ((self.reaches['reaches'][reach]['length'] * incision_depth) * 2)
        self.reaches['reaches'][reach]['width'] = self.reaches['reaches'][reach]['width'] + dw

        # adjust elevation
        dz = wall_vol / (self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['bankfull_width'])
        self.reaches['reaches'][reach]['elevation'] = self.reaches['reaches'][reach]['elevation'] + dz

    def simulation(self):
        transfer_vals = {
            "upstream": {"Qs_out": None},
            "deposit_upstream": {"Qs_in": None, "Qs_out": None},
            "deposit_downstream": {"Qs_in": None, "Qs_out": None},
            "downstream": {"Qs_in": None, "Qs_out": None}
        }

        for i in tqdm(self.q.index):
            if i == 47:
                print(i)
            ts = self.q.loc[i, 'Datetime']
            self.reaches['timestep'] = ts
            q_in = self.q.loc[i, 'Q'] * self.boundary_conditions['flow_scale']
            width_in = self.w_geom[0]*self.boundary_conditions['flow_scale'] * q_in ** self.w_geom[1]
            if width_in > self.boundary_conditions['bankfull_width']:
                width_in = self.boundary_conditions['bankfull_width']
            depth_in = self.h_geom[0]*self.boundary_conditions['flow_scale'] * q_in ** self.h_geom[1]
            fractions_in = {float(s): f for s, f in self.boundary_conditions['upstream_gsd'].items()}
            qs_fractions = transport(fractions_in, self.boundary_conditions['upstream_slope'], q_in,
                                        depth_in, width_in, self.time_interval)
            transfer_vals['upstream']['Qs_in'] = qs_fractions

            for reach, attributes in self.reaches['reaches'].items():
                if reach == 'upstream':
                    qs_in = transfer_vals['upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = self.w_geom[0]*attributes['flow_scale'] * q_in ** self.w_geom[1]
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    depth_in = self.h_geom[0]*attributes['flow_scale'] * q_in ** self.h_geom[1]
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_upstream']['elevation']) / \
                            attributes['length']

                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    d50, d84 = percentiles(attributes['gsd'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('upstream', ts), 'D50'] = d50 * 1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    fractions_in = update_fractions_in(fractions, qs_in, active_volume, 0.21)

                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval)
                    transfer_vals['upstream']['Qs_out'] = qs_fractions
                    transfer_vals['deposit_upstream']['Qs_in'] = qs_fractions
                    self.out_df.loc[('upstream', ts), 'yield'] = sum([trans[1] for size, trans in qs_fractions.items()])
                    self.reaches['reaches']['upstream']['Qs_in'] = sum([trans[1] for size, trans in qs_in.items()])
                    self.reaches['reaches']['upstream']['Qs_out'] = sum([trans[1] for size, trans in qs_fractions.items()])

                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_fractions)
                    dz = exner(d_qs_bed, attributes['length'], attributes['bankfull_width'], 0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['upstream']['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[('upstream', ts), 'elev'] = attributes['elevation'] + dz

                    # update fractions
                    d50, d84 = percentiles(attributes['gsd'])
                    d50, d84 = (2**-d50)/1000, (2**-d84)/1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    self.reaches['reaches']['upstream']['gsd'] = update_fractions_out(fractions_in, qs_fractions,
                                                                                      active_volume, 0.21)

                if reach == 'deposit_upstream':
                    qs_in = transfer_vals['deposit_upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = attributes['width']
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    depth_in = self.h_geom[0]*attributes['flow_scale'] * q_in ** self.h_geom[1]
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_downstream']['elevation']) / \
                            attributes['length']

                    fractions = {float(s): f for s, f in attributes['gsd_bed'].items()}
                    d50, d84 = percentiles(attributes['gsd_bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('deposit_upstream', ts), 'D50'] = d50 * 1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    fractions_updated = update_fractions_in(fractions, qs_in, active_volume, 0.21)
                    fractions_in = {'bed': {float(s): f for s, f in fractions_updated.items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}

                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                             twod=True, lwd_factor=1)
                    transfer_vals['deposit_upstream']['Qs_out'] = qs_fractions
                    transfer_vals['deposit_downstream']['Qs_in'] = qs_fractions
                    self.out_df.loc[('deposit_upstream', ts), 'yield'] = \
                        sum([trans[1] for size, trans in qs_fractions['bed'].items()]) + \
                        sum([trans[1] for size, trans in qs_fractions['wall'].items()])
                    self.reaches['reaches']['deposit_upstream']['Qs_in'] = sum([trans[1] for size, trans in qs_in.items()])
                    self.reaches['reaches']['deposit_upstream']['Qs_out_bed'] = sum([trans[1] for size, trans in
                                                                         qs_fractions['bed'].items()])
                    self.reaches['reaches']['deposit_upstream']['Qs_out_wall'] = sum([trans[1] for size, trans in
                                                                                     qs_fractions['wall'].items()])

                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_upstream']['Qs_in'],
                                                                transfer_vals['deposit_upstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'],
                                    0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['deposit_upstream']['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, 0.21)
                    self.reaches['reaches']['deposit_upstream']['width'] = attributes['width'] + dw
                    self.out_df.loc[('deposit_upstream', ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[('deposit_upstream', ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    d50_bed, d84_bed = percentiles(attributes['gsd_bed'])
                    d50_bed, d84_bed = (2 ** -d50_bed) / 1000, (2 ** -d84_bed) / 1000
                    active_volume_bed = attributes['length'] * attributes['bankfull_width'] * d50_bed
                    d50_wall, d84_wall = percentiles(attributes['gsd_wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000
                    active_volume_wall = attributes['length'] * depth_in * (d50_wall*2)
                    self.reaches['reaches']['deposit_upstream']['gsd_bed'] = update_fractions_out(
                        fractions_in['bed'], qs_fractions['bed'], active_volume_bed, 0.21)
                    self.reaches['reaches']['deposit_upstream']['gsd_wall'] = update_fractions_out(
                        fractions_in['wall'], qs_fractions['wall'], active_volume_wall, 0.21)

                    incision = self.init_elevs['deposit_upstream'] - attributes['elevation'] + dz
                    angle = atan(incision / attributes['width'] + dw)
                    if angle > 0.35:  # if the angle to the center of channel is 20 degrees its probably 25-30 at bank
                        self.incision_feedback("deposit_upstream", incision)
                        logging.info(f'bank slumping feedback at timestep {i}')

                if reach == 'deposit_downstream':
                    qs_in = transfer_vals['deposit_downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = attributes['width']
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    depth_in = self.h_geom[0] * attributes['flow_scale'] * q_in ** self.h_geom[1]
                    slope = (attributes['elevation'] - self.reaches['reaches']['downstream']['elevation']) / \
                            attributes['length']

                    fractions = {float(s): f for s, f in attributes['gsd_bed'].items()}
                    d50, d84 = percentiles(attributes['gsd_bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('deposit_downstream', ts), 'D50'] = d50 * 1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    fractions_updated = update_fractions_in(fractions, qs_in, active_volume, 0.21)
                    fractions_in = {'bed': {float(s): f for s, f in fractions_updated.items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}

                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                             twod=True, lwd_factor=1)
                    transfer_vals['deposit_downstream']['Qs_out'] = qs_fractions
                    transfer_vals['downstream']['Qs_in'] = qs_fractions
                    self.out_df.loc[('deposit_downstream', ts), 'yield'] = \
                        sum([trans[1] for size, trans in qs_fractions['bed'].items()]) + \
                        sum([trans[1] for size, trans in qs_fractions['wall'].items()])
                    self.reaches['reaches']['deposit_downstream']['Qs_in'] = sum([trans[1] for size, trans in
                                                                                qs_in['bed'].items()]) + \
                                                                           sum([trans[1] for size, trans in
                                                                                qs_in['wall'].items()])
                    self.reaches['reaches']['deposit_downstream']['Qs_out_bed'] = sum([trans[1] for size, trans in
                                                                                 qs_fractions['bed'].items()])
                    self.reaches['reaches']['deposit_downstream']['Qs_out_wall'] = sum([trans[1] for size, trans in
                                                                                       qs_fractions['wall'].items()])

                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_downstream']['Qs_in'],
                                                                transfer_vals['deposit_downstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'],
                               0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['deposit_downstream']['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, 0.21)
                    self.reaches['reaches']['deposit_downstream']['width'] = attributes['width'] + dw
                    self.out_df.loc[('deposit_downstream', ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[('deposit_downstream', ts), 'width'] = attributes['width'] + dw

                    #logging.info(f"incision: {self.init_elevs['deposit_downstream'] - attributes['elevation'] + dz}; "
                    #             f"\widening: {attributes['width'] + dw - 1}")

                    # update fractions
                    d50_bed, d84_bed = percentiles(attributes['gsd_bed'])
                    d50_bed, d84_bed = (2 ** -d50_bed) / 1000, (2 ** -d84_bed) / 1000
                    active_volume_bed = attributes['length'] * attributes['bankfull_width'] * d50_bed
                    d50_wall, d84_wall = percentiles(attributes['gsd_wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000
                    active_volume_wall = attributes['length'] * depth_in * (d50_wall * 2)
                    self.reaches['reaches']['deposit_downstream']['gsd_bed'] = update_fractions_out(
                        fractions_in['bed'], qs_fractions['bed'], active_volume_bed, 0.21)
                    self.reaches['reaches']['deposit_downstream']['gsd_wall'] = update_fractions_out(
                        fractions_in['wall'], qs_fractions['wall'], active_volume_wall, 0.21)

                    incision = self.init_elevs['deposit_downstream'] - attributes['elevation'] + dz
                    angle = atan(incision / attributes['width'] + dw)
                    if angle > 0.35:  # if the angle to the center of channel is 20 degrees its probably 25-30 at bank
                        self.incision_feedback("deposit_downstream", incision)

                if reach == 'downstream':
                    qs_in = transfer_vals['downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = self.w_geom[0]*attributes['flow_scale'] * q_in ** self.w_geom[1]
                    if width_in > attributes['bankfull_width']:
                        width_in = attributes['bankfull_width']
                    depth_in = self.h_geom[0]*attributes['flow_scale'] * q_in ** self.h_geom[1]
                    slope = self.boundary_conditions['downstream_slope']

                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    d50, d84 = percentiles(attributes['gsd'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('downstream', ts), 'D50'] = d50 * 1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    fractions_in = update_fractions_in(fractions, qs_in, active_volume, 0.21)

                    qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                             lwd_factor=1)
                    transfer_vals['downstream']['Qs_out'] = qs_fractions
                    self.out_df.loc[('downstream', ts), 'yield'] = sum([trans[1] for size, trans in qs_fractions.items()])
                    self.reaches['reaches']['downstream']['Qs_in'] = sum([trans[1] for size, trans in
                                                                                qs_in['bed'].items()]) + \
                                                                           sum([trans[1] for size, trans in
                                                                                qs_in['wall'].items()])
                    self.reaches['reaches']['downstream']['Qs_out'] = sum([trans[1] for size, trans in qs_fractions.items()])

                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_fractions)
                    dz = exner(d_qs_bed, attributes['length'], attributes['bankfull_width'],
                               0.21)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['downstream']['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[('downstream', ts), 'elev'] = attributes['elevation'] + dz

                    # update fractions
                    d50, d84 = percentiles(attributes['gsd'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    active_volume = attributes['length'] * attributes['bankfull_width'] * d50
                    self.reaches['reaches']['downstream']['gsd'] = update_fractions_out(fractions_in, qs_fractions,
                                                                                      active_volume, 0.21)

            #if i in [10, 50, 500, 1000, 1500]:
            #    self.serialize_timestep(f'../Outputs/{self.reach_name}_{i}.json')

        self.save_df()



b_c = '../Inputs/boundary_conditions.json'
r_a = '../Inputs/reaches.json'
dis = '../Inputs/Woods_Q_2012_2019_1hr.csv'
t_i = 3600
whg = [5.947, 0.115]
dhg = [0.283, 0.402]
r_n = 'Woods'

inst = DepositEvolution(b_c, r_a, dis, t_i, whg, dhg, r_n)
inst.simulation()
