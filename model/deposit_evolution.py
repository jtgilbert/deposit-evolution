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
    """
    Calculate change in elevation using Exner's equation
    :param d_qs_bed:
    :param reach_length:
    :param reach_width:
    :param porosity:
    :return:
    """
    volume = d_qs_bed / ((1 - porosity) * 2650)
    dz = volume / (reach_length * reach_width)

    return dz


def delta_width(d_qs_wall, reach_length, depth, porosity):
    """
    Calculate change in width
    :param d_qs_wall:
    :param reach_length:
    :param depth:
    :param porosity:
    :return:
    """

    volume = d_qs_wall / ((1 - porosity) * 2650)
    dw = volume / ((reach_length * depth) * 2)

    return dw


def percentiles(fractions):
    """
    Calculates the D50 and D84 from input size fractions
    :param fractions:
    :return:
    """
    out_sizes = []

    if fractions[1.0] > 0.84:
        out_sizes.append(-0.5)
        out_sizes.append(0.5)
        return out_sizes[0], out_sizes[1]

    elif 0.5 < fractions[1.0] < 0.84:
        out_sizes.append(-0.5)
        cumulative = 0
        grain_size = None
        for size, fraction in fractions.items():
            if cumulative < 0.84:
                cumulative += fraction
                grain_size = float(size)
        for i, size in enumerate(fractions.keys()):
            if float(size) == grain_size:
                low_size = float(list(fractions.keys())[i - 1]) - 0.25
                low_cum = cumulative - float(fractions[list(fractions.keys())[i - 1]])
                high_size = grain_size - 0.25
                high_cum = cumulative
                p_out = ((low_cum / 0.84) * low_size + (0.84 / high_cum) * high_size) / ((low_cum / 0.84) + (0.84 / high_cum))
                out_sizes.append(p_out)
        return out_sizes[0], out_sizes[1]

    else:
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
    """
    Find the change in each size fraction based on inputs and outputs of that fraction
    :param qsinputs:
    :param qsoutputs:
    :return:
    """

    if len(qsinputs.keys()) != 2:
        tot_in = sum(qsinputs.values())
    else:
        tot_in = sum(qsinputs['bed'].values()) + sum(qsinputs['wall'].values())

    if len(qsoutputs.keys()) != 2:
        tot_out = sum(qsoutputs.values())
        d_qs_bed = tot_in - tot_out
        d_qs_wall = None

    else:
        tot_bed = sum(qsoutputs['bed'].values())
        d_qs_wall = sum(qsoutputs['wall'].values())
        tot_out = tot_bed + d_qs_wall
        d_qs_bed = tot_in - tot_bed

    return d_qs_bed, d_qs_wall


def update_fractions_out(fractions, qs_out, active_volume, porosity, minimum_frac=None):
    """
    Basically the same as update bed fractions but applied to wall fractions
    :param fractions:
    :param qs_out:
    :param active_volume:
    :param porosity:
    :param minimum_frac:
    :return:
    """
    # run separate for bed and wall fractions
    existing_mass = {phi: frac * active_volume * (1-porosity) * 2650 for phi, frac in fractions.items()}

    new_mass = {phi: existing_mass[phi] - qs_out[phi] for phi in existing_mass.keys()}

    fractions_out = {phi: new_mass[phi] / sum(new_mass.values()) for phi in new_mass.keys()}

    if minimum_frac:
        counter = 0
        tot_added = 0
        ex_vals = [f for p, f in fractions_out.items()]
        ex_vals.sort()
        for phi, fract in fractions_out.items():
            if float(phi) >= -6 and fract < minimum_frac:
                counter += 1
                add = minimum_frac - fract
                tot_added += add
                fractions_out[phi] = fract + add
            if float(phi) < -6 and 0 < fract < minimum_frac:
                counter += 1
                add = minimum_frac - fract
                tot_added += add
                fractions_out[phi] = fract + add
        if counter > 0:
            vals_to_subtr = ex_vals[-counter:]
            for val in vals_to_subtr:
                if val <= minimum_frac:
                    vals_to_subtr.remove(val)
                    counter -= 1
            subtract = tot_added / counter
            for phi, fract in fractions_out.items():
                if fract in vals_to_subtr:
                    fractions_out[phi] = fract - subtract

    for key, val in fractions_out.items():
        if val < 0:
            raise Exception('Fraction less than 0')
        # if val < 0.001 and key > -7:
        #     raise Exception('fraction less than 0.001')

    total = sum([frac for size, frac in fractions_out.items()])
    if total > 1.1:
        raise Exception('Fractions sum to more than 1')
    if total < 0.95:
        raise Exception('Fractions sum to less than 1')

    return fractions_out


def update_bed_fractions(fractions, qs_in, qs_out, active_volume, porosity, minimum_frac=None):
    """
    Update fractions of each size class in the bed. Combines existing fractions with difference between
    transport of each fraction into and out of the reach.
    :param fractions:
    :param qs_in:
    :param qs_out:
    :param active_volume:
    :param porosity:
    :param minimum_frac:
    :return:
    """

    # get existin mass of each fraction in active layer
    existing_mass = {phi: frac * active_volume * (1-porosity) * 2650 for phi, frac in fractions.items()}

    # get change in mass for each fraction
    d_qs = {phi: qs_in[phi] - qs_out[phi] for phi in qs_in.keys()}

    # find the new mass of each fraction and convert to new fractions
    new_mass = {phi: existing_mass[phi] + d_qs[phi] for phi in existing_mass.keys()}
    if sum(new_mass.values()) < 1e-10:
        #logger.info(f'')
        raise Exception('new mass getting too small')
    fractions_out = {phi: new_mass[phi] / sum(new_mass.values()) for phi in new_mass.keys()}

    if minimum_frac:
        counter = 0
        tot_added = 0
        ex_vals = [f for p, f in fractions_out.items()]
        ex_vals.sort()
        for phi, fract in fractions_out.items():
            if float(phi) >= -6 and fract < minimum_frac:
                counter += 1
                add = minimum_frac - fract
                tot_added += add
                fractions_out[phi] = fract + add
            if float(phi) < -6 and 0 < fract < minimum_frac:
                counter += 1
                add = minimum_frac - fract
                tot_added += add
                fractions_out[phi] = fract + add
        if counter > 0:
            vals_to_subtr = ex_vals[-counter:]
            for val in vals_to_subtr:
                if val <= minimum_frac:
                    vals_to_subtr.remove(val)
                    counter -= 1
            subtract = tot_added / counter
            for phi, fract in fractions_out.items():
                if fract in vals_to_subtr:
                    fractions_out[phi] = fract - subtract

    for key, val in fractions_out.items():
        if val < 0:
            raise Exception('Fraction less than 0')
        # if val < 0.001 and key > -7:
        #     raise Exception('fraction less than 0.001')

    total = sum([frac for size, frac in fractions_out.items()])
    if total > 1.1:
        raise Exception('Fractions sum to more than 1')
    if total < 0.95:
        raise Exception('Fractions sum to less than 1')

    return fractions_out


class DepositEvolution:
    def __init__(self, bound_conds: str, reach_atts: str, discharge: str, time_interval: int, width_hydro_geom: List,
                 depth_hydro_geom: List, v_hydro_geom: List, reach_name: str):
        """
        The deposit evolution model class
        :param bound_conds:
        :param reach_atts:
        :param discharge:
        :param time_interval:
        :param width_hydro_geom:
        :param depth_hydro_geom:
        :param v_hydro_geom:
        :param reach_name:
        """

        f_bc = open(bound_conds)
        self.boundary_conditions = json.load(f_bc)
        f_ra = open(reach_atts)
        self.reaches = json.load(f_ra)
        self.w_geom = width_hydro_geom
        self.h_geom = depth_hydro_geom
        self.v_geom = v_hydro_geom
        self.reach_name = reach_name
        self.porosity = self.boundary_conditions['sediment_porosity']

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
        zeros = np.zeros((len(reach) * len(timestep), 5))
        self.out_df = pd.DataFrame(zeros, index=index, columns=['elev', 'width', 'D50', 'slope', 'yield'])

        # get initial active layer thickness and min elev for each reach
        us_gsd = {float(s): f for s, f in self.boundary_conditions['reaches']['upstream']['gsd'].items()}
        us_d50, us_d84 = percentiles(us_gsd)
        us_thickness = 2 * ((2**-us_d84) / 1000)
        us_minelev = self.reaches['reaches']['upstream']['elevation'] - us_thickness
        df_us_minelev = self.boundary_conditions['reaches']['deposit_upstream']['pre_elev'] - us_thickness
        ds_gsd = {float(s): f for s, f in self.boundary_conditions['reaches']['downstream']['gsd'].items()}
        ds_d50, ds_d84 = percentiles(ds_gsd)
        ds_thickness = 2 * ((2 ** -ds_d84) / 1000)
        ds_minelev = self.reaches['reaches']['downstream']['elevation'] - ds_thickness
        df_ds_minelev = self.boundary_conditions['reaches']['deposit_downstream']['pre_elev'] - ds_thickness

        self.boundary_conditions['min_elev'] = {'upstream': us_minelev, 'deposit_upstream': df_us_minelev,
                                                 'deposit_downstream': df_ds_minelev, 'downstream': ds_minelev}

        self.active_layer = {'upstream': us_thickness, 'deposit_upstream': us_thickness,
                             'deposit_downstream': ds_thickness, 'downstream': ds_thickness}

        self.init_elevs = {key: self.reaches['reaches'][key]['elevation'] for key in self.reaches['reaches'].keys()}

    def serialize_timestep(self, outfile):
        """
        Save a .json file of a given time step
        :param outfile:
        :return:
        """

        with open(outfile, 'w') as dst:
            json.dump(self.reaches, dst, indent=4)

    def save_df(self):
        """
        Save the output data frame to a csv file
        :return:
        """
        logging.info('Saving output csv')
        self.out_df.to_csv(f'../Outputs/{self.reach_name}_out.csv')

    def incision_feedback(self, reach, incision_depth, active_vol_bed, active_vol_wall, ts):
        """
        Simulate bank failure resulting from incision
        :param reach:
        :param incision_depth:
        :param active_vol_bed:
        :param active_vol_wall:
        :param ts:
        :return:
        """
        # look at updating wall gsd because it only uses active volume not full incision depth
        d50_bed, d84_bed = percentiles(self.reaches['reaches'][reach]['gsd_bed'])
        d50_bed, d84_bed = 2**-d50_bed / 1000, 2**-d84_bed / 1000
        d50_wall, d84_wall = percentiles(self.reaches['reaches'][reach]['gsd_wall'])
        d50_wall, d84_wall = 2**-d50_wall / 1000, 2**-d84_wall / 1000

        # bed_vol = d50_bed * self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['bankfull_width']
        # wall_vol = d50_wall * self.reaches['reaches'][reach]['length'] * incision_depth

        tot_vol = active_vol_bed + active_vol_wall

        # new bed gsd
        self.reaches['reaches'][reach]['gsd_bed'] = \
            {key: ((self.reaches['reaches'][reach]['gsd_bed'][key]*active_vol_bed) +
                  (self.reaches['reaches'][reach]['gsd_wall'][key]*active_vol_wall)) /
                  tot_vol for key in self.reaches['reaches'][reach]['gsd_bed'].keys()}

        # new wall gsd
        bound_gsd = {float(s): frac for s, frac in self.boundary_conditions['deposit_gsd'].items()}
        self.reaches['reaches'][reach]['gsd_wall'] = \
            {key: ((self.reaches['reaches'][reach]['gsd_wall'][key] * (0.5*active_vol_wall)) +
                  (bound_gsd[key] * (0.5*active_vol_wall))) /
                  active_vol_wall for key in self.reaches['reaches'][reach]['gsd_wall'].keys()}

        # adjust width
        dw = active_vol_wall / ((self.reaches['reaches'][reach]['length'] * incision_depth) * 2)
        self.reaches['reaches'][reach]['width'] = self.reaches['reaches'][reach]['width'] + dw
        self.out_df.loc[(reach, ts), 'width'] = self.reaches['reaches'][reach]['width'] + dw

        # adjust elevation
        dz = active_vol_wall / (self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['width'])
        self.reaches['reaches'][reach]['elevation'] = self.reaches['reaches'][reach]['elevation'] + dz
        self.out_df.loc[(reach, ts), 'elev'] = self.reaches['reaches'][reach]['elevation'] + dz

    def simulation(self):
        """
        Run the simulation
        :return:
        """

        transfer_vals = {
            "upstream": {"Qs_out": None},
            "deposit_upstream": {"Qs_in": None, "Qs_out": None},
            "deposit_downstream": {"Qs_in": None, "Qs_out": None},
            "downstream": {"Qs_in": None, "Qs_out": None}
        }

        dep_gsd = {float(s): f for s, f in self.boundary_conditions['deposit_gsd'].items()}

        tot_in = 0.0

        for i in tqdm(self.q.index):
            trigger = False
            ts = self.q.loc[i, 'Datetime']
            if ts == '5/17/2021 23:00':
                print('checking peak flow')
            self.reaches['timestep'] = ts
            q_in = self.q.loc[i, 'Q'] * self.boundary_conditions['flow_scale']
            width_in = self.w_geom[0] * q_in ** self.w_geom[1]
            if width_in > self.boundary_conditions['bankfull_width']:
                width_in = self.boundary_conditions['bankfull_width']
            depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
            fractions_in = {float(s): f for s, f in self.boundary_conditions['upstream_gsd'].items()}
            slope = self.boundary_conditions['upstream_slope']
            qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval)
            qs_kg = {key: val[1] for key, val in qs_fractions.items()}
            tot_in += sum(qs_kg.values())
            logging.info(f'{ts} Qs_in: {sum(qs_kg.values())}')
            transfer_vals['upstream']['Qs_in'] = qs_kg

            for reach, attributes in self.reaches['reaches'].items():
                if reach == 'upstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd'].keys()) == 0:
                        self.reaches['reaches']['upstream']['gsd'] = self.boundary_conditions['reaches']['upstream']['gsd']
                        attributes['gsd'] = self.boundary_conditions['reaches']['upstream']['gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_upstream']['elevation']) / \
                            attributes['length']
                    self.out_df.loc[('upstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    d50, d84 = percentiles(fractions)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('upstream', ts), 'D50'] = d50 * 1000
                    if d50 > 0.002:
                        tau_star = (9810 * depth_in * slope) / (1650 * 9.81 * d50)
                        active_layer = 7968 * tau_star**2.61 * d50  # from Wilcock 1997
                        if active_layer > 2 * d84:
                            active_layer = 2 * d84
                    else:
                        active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev']['upstream']:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev']['upstream']
                    active_volume = attributes['length'] * width_in * active_layer

                    # fractions_in = update_fractions_in(fractions, qs_in, active_volume, 0.21)
                    self.reaches['reaches']['upstream']['D50'] = d50
                    self.reaches['reaches']['upstream']['D84'] = d84
                    # if d84/d50 < 1.5 or d84/d50 > 10:
                    #     trigger = True

                    # calculate transport
                    if slope < 0:
                        qs_fractions = {key: [0., 0.] for key in fractions.keys()}
                    else:
                        qs_fractions = transport(fractions, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: max(0, (frac - 0.0025)) * active_volume * (1 - self.porosity) * 2650 for key, frac in
                                    fractions.items()}
                    for size, frac in qs_kg.items():
                        if frac > qs_in[size] + max_bed_recr[size]:
                            qs_kg[size] = qs_in[size] + max_bed_recr[size]

                    transfer_vals['upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_upstream']['Qs_in'] = qs_kg
                    self.out_df.loc[('upstream', ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches']['upstream']['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches']['upstream']['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], width_in, self.porosity)  # porosity (0.21) from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['upstream']['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[('upstream', ts), 'elev'] = attributes['elevation'] + dz

                    # update bed fractions
                    if active_volume > 0:
                        self.reaches['reaches']['upstream']['gsd'] = update_bed_fractions(fractions, qs_in, qs_kg, active_volume, self.porosity, minimum_frac=0.003)
                    else:
                        self.reaches['reaches']['upstream']['gsd'] = self.boundary_conditions['reaches']['upstream']['gsd']  # fractions

                if reach == 'deposit_upstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd_bed'].keys()) == 0:
                        self.reaches['reaches']['deposit_upstream']['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        self.reaches['reaches']['deposit_upstream']['gsd_wall'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_wall'] = self.boundary_conditions['deposit_gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['deposit_upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = min(attributes['width'], self.w_geom[0] * q_in ** self.w_geom[1])
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    v_in = self.v_geom[0] * q_in ** self.v_geom[1]
                    depth_in = q_in / (width_in * v_in)
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_downstream']['elevation']) / \
                            attributes['length']
                    self.out_df.loc[('deposit_upstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {'bed': {float(s): f for s, f in attributes['gsd_bed'].items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                    d50, d84 = percentiles(fractions_in['bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('deposit_upstream', ts), 'D50'] = d50 * 1000
                    if d50 > 0.002:
                        tau_star = (9810 * depth_in * slope) / (1650 * 9.81 * d50)
                        active_layer = 7968 * tau_star ** 2.61 * d50  # from Wilcock 1997
                        if active_layer > 2 * d84:
                            active_layer = 2 * d84
                    else:
                        active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev']['deposit_upstream']:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev']['deposit_upstream']

                    d50_wall, d84_wall = percentiles(fractions_in['wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000
                    if d50_wall > 0.002:
                        active_layer_wall = d50_wall
                    else:
                        active_layer_wall = 2 * d50_wall

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * active_layer_wall * 2
                    # fractions_updated = update_fractions_in(fractions, qs_in, active_volume, 0.21)
                    self.reaches['reaches']['deposit_upstream']['D50'] = d50
                    self.reaches['reaches']['deposit_upstream']['D84'] = d84
                    # if d84/d50 < 1.5 or d84/d50 > 10:
                    #     trigger = True

                    # calculate transport
                    if slope < 0:
                        qs_fractions = {'bed': {key: [0., 0.] for key in fractions_in['bed'].keys()},
                                        'wall': {key: [0., 0.] for key in fractions_in['wall'].keys()}}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                    qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                             'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: max(0, (frac - 0.0025)) * active_volume * (1 - self.porosity) * 2650 for key, frac in
                                    fractions_in['bed'].items()}
                    max_wall_recr = {key: max(0, (frac - 0.0025)) * active_volume_wall * (1 - self.porosity) * 2650 for key, frac
                                     in
                                     fractions_in['wall'].items()}

                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in[size] + max_bed_recr[size]:
                            qs_kg['bed'][size] = qs_in[size] + max_bed_recr[size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > max_wall_recr[size]:
                            qs_kg['wall'][size] = max_wall_recr[size]

                    transfer_vals['deposit_upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_downstream']['Qs_in'] = qs_kg
                    self.out_df.loc[('deposit_upstream', ts), 'yield'] = sum(qs_kg['bed'].values()) + \
                        sum(qs_kg['wall'].values())
                    self.reaches['reaches']['deposit_upstream']['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches']['deposit_upstream']['Qs_out_bed'] = sum(qs_kg['bed'].values())
                    self.reaches['reaches']['deposit_upstream']['Qs_out_wall'] = sum(qs_kg['wall'].values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_upstream']['Qs_in'],
                                                                transfer_vals['deposit_upstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['deposit_upstream']['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches']['deposit_upstream']['width'] = attributes['width'] + dw
                    self.out_df.loc[('deposit_upstream', ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[('deposit_upstream', ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    if active_volume > 0:
                        self.reaches['reaches']['deposit_upstream']['gsd_bed'] = update_bed_fractions(
                            fractions_in['bed'], qs_in, qs_kg['bed'], active_volume, self.porosity, minimum_frac=0.003)
                    else:
                        self.reaches['reaches']['deposit_upstream']['gsd_bed'] = self.boundary_conditions['reaches']['deposit_upstream']['gsd'] # fractions_in['bed']
                    if active_volume_wall > 0:
                        self.reaches['reaches']['deposit_upstream']['gsd_wall'] = update_fractions_out(
                            fractions_in['wall'], qs_kg['wall'], active_volume_wall, self.porosity, minimum_frac=0.003)

                    # if the channel incises, expose new fractions in the walls
                    if dz < 0:
                        for size, frac in self.reaches['reaches']['deposit_upstream']['gsd_wall'].items():
                            dep_frac = dep_gsd[size]
                            new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                            # new_frac = frac * (depth_in / (depth_in + abs(dz))) + dep_frac * (abs(dz)/(depth_in + abs(dz)))
                            new_frac = (frac * active_volume_wall + dep_frac * (attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                            self.reaches['reaches']['deposit_upstream']['gsd_wall'][size] = new_frac

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs['deposit_upstream'] - attributes['elevation'] + dz
                    angle = atan(incision / (attributes['width'] + dw))
                    if angle > 0.2 and active_volume > 0:  # if the angle to the center of channel is 20 degrees its probably 25-30 at bank
                        self.incision_feedback("deposit_upstream", incision, active_volume, active_volume_wall, ts)
                        logging.info(f'bank slumping feedback for deposit upstream at timestep {i}')

                if reach == 'deposit_downstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd_bed'].keys()) == 0:
                        self.reaches['reaches']['deposit_downstream']['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        self.reaches['reaches']['deposit_downstream']['gsd_wall'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_wall'] = self.boundary_conditions['deposit_gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['deposit_downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = min(attributes['width'], self.w_geom[0] * q_in ** self.w_geom[1])
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    v_in = self.v_geom[0] * q_in ** self.v_geom[1]
                    depth_in = q_in / (width_in * v_in)
                    slope = (attributes['elevation'] - self.reaches['reaches']['downstream']['elevation']) / \
                            attributes['length']
                    self.out_df.loc[('deposit_downstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {'bed': {float(s): f for s, f in attributes['gsd_bed'].items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                    d50, d84 = percentiles(fractions_in['bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('deposit_downstream', ts), 'D50'] = d50 * 1000

                    if d50 > 0.002:
                        tau_star = (9810 * depth_in * slope) / (1650 * 9.81 * d50)
                        active_layer = 7968 * tau_star ** 2.61 * d50  # from Wilcock 1997
                        if active_layer > 2 * d84:
                            active_layer = 2 * d84
                    else:
                        active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev']['deposit_downstream']:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev']['deposit_downstream']

                    d50_wall, d84_wall = percentiles(fractions_in['wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000
                    if d50_wall > 0.002:
                        active_layer_wall = d50_wall
                    else:
                        active_layer_wall = 2 * d50_wall

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * (active_layer_wall * 2)
                    # fractions_updated = update_fractions_in(fractions, qs_in, active_volume, 0.21)
                    # fractions_in = {'bed': {float(s): f for s, f in fractions.items()},
                    #                 'wall': {float(s): f for s, f in fractions_wall.items()}}
                    self.reaches['reaches']['deposit_downstream']['D50'] = d50
                    self.reaches['reaches']['deposit_downstream']['D84'] = d84
                    # if d84/d50 < 1.5 or d84/d50 > 10:
                    #     trigger = True

                    # calculate transport
                    if slope < 0:
                        qs_fractions = {'bed': {key: [0., 0.] for key in fractions_in['bed'].keys()},
                                        'wall': {key: [0., 0.] for key in fractions_in['wall'].keys()}}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                    qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                             'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: max(0, (frac - 0.0025)) * active_volume * (1 - self.porosity) * 2650 for key, frac in
                                    fractions_in['bed'].items()}
                    max_wall_recr = {key: max(0, (frac - 0.0025)) * active_volume_wall * (1 - self.porosity) * 2650 for key, frac
                                     in fractions_in['wall'].items()}
                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]:
                            qs_kg['bed'][size] = qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > max_wall_recr[size]:
                            qs_kg['wall'][size] = max_wall_recr[size]

                    transfer_vals['deposit_downstream']['Qs_out'] = qs_kg
                    transfer_vals['downstream']['Qs_in'] = qs_kg
                    self.out_df.loc[('deposit_downstream', ts), 'yield'] = sum(qs_kg['bed'].values()) + \
                                                                           sum(qs_kg['wall'].values())
                    self.reaches['reaches']['deposit_downstream']['Qs_in'] = sum(qs_in['bed'].values()) + \
                                                                             sum(qs_in['wall'].values())
                    self.reaches['reaches']['deposit_downstream']['Qs_out_bed'] = sum(qs_kg['bed'].values())
                    self.reaches['reaches']['deposit_downstream']['Qs_out_wall'] = sum(qs_kg['wall'].values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_downstream']['Qs_in'],
                                                                transfer_vals['deposit_downstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['deposit_downstream']['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches']['deposit_downstream']['width'] = attributes['width'] + dw
                    self.out_df.loc[('deposit_downstream', ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[('deposit_downstream', ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    if active_volume > 0:
                        self.reaches['reaches']['deposit_downstream']['gsd_bed'] = update_bed_fractions(
                            fractions_in['bed'], tot_qs_in, qs_kg['bed'], active_volume, self.porosity, minimum_frac=0.003)
                    else:
                        self.reaches['reaches']['deposit_downstream']['gsd_bed'] = self.boundary_conditions['reaches']['deposit_downstream']['gsd'] # fractions_in['bed']
                    if active_volume_wall > 0:
                        self.reaches['reaches']['deposit_downstream']['gsd_wall'] = update_fractions_out(
                            fractions_in['wall'], qs_kg['wall'], active_volume_wall, self.porosity, minimum_frac=0.003)

                    # if the channel incises, expose new fractions in the walls
                    if dz < 0:
                        for size, frac in self.reaches['reaches']['deposit_downstream']['gsd_wall'].items():
                            dep_frac = dep_gsd[size]
                            new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                            # new_frac = frac * (depth_in / (depth_in + abs(dz))) + dep_frac * (abs(dz)/(depth_in + abs(dz)))
                            new_frac = (frac * active_volume_wall + dep_frac * (
                                        attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                            self.reaches['reaches']['deposit_downstream']['gsd_wall'][size] = new_frac

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs['deposit_downstream'] - attributes['elevation'] + dz
                    angle = atan(incision / (attributes['width'] + dw))
                    if angle > 0.2 and active_volume > 0:
                        self.incision_feedback("deposit_downstream", incision, active_volume, active_volume_wall, ts)
                        logging.info(f'bank slumping feedback for deposit downstream at timestep {i}')

                if reach == 'downstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd'].keys()) == 0:
                        self.reaches['reaches']['downstream']['gsd'] = self.boundary_conditions['reaches']['downstream']['gsd']
                        attributes['gsd'] = self.boundary_conditions['reaches']['downstream']['gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    if width_in > attributes['bankfull_width']:
                        width_in = attributes['bankfull_width']
                    depth_in = self.h_geom[0] * q_in ** self.h_geom[1]
                    slope = (attributes['elevation'] - self.boundary_conditions['downstream_elev']) / attributes['length']
                    self.out_df.loc[('downstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    d50, d84 = percentiles(fractions)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[('downstream', ts), 'D50'] = d50 * 1000
                    if d50 > 0.002:
                        tau_star = (9810 * depth_in * slope) / (1650 * 9.81 * d50)
                        active_layer = 7968 * tau_star ** 2.61 * d50  # from Wilcock 1997
                        if active_layer > 2 * d84:
                            active_layer = 2 * d84
                    else:
                        active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev']['downstream']:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev']['downstream']
                    active_volume = attributes['length'] * width_in * active_layer

                    self.reaches['reaches']['downstream']['D50'] = d50
                    self.reaches['reaches']['downstream']['D84'] = d84
                    # if d84/d50 < 1.5 or d84/d50 > 10:
                    #     trigger = True

                    # calculate transport
                    if slope < 0:
                        qs_fractions = {key: [0., 0.] for key in fractions.keys()}
                    else:
                        qs_fractions = transport(fractions, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: max(0, (frac - 0.0025)) * active_volume * (1 - self.porosity) * 2650 for key, frac in
                                    fractions.items()}

                    for size, frac in qs_kg.items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]:
                            qs_kg[size] = qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]

                    transfer_vals['downstream']['Qs_out'] = qs_kg
                    self.out_df.loc[('downstream', ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches']['downstream']['Qs_in'] = sum(qs_in['bed'].values()) + \
                                                                     sum(qs_in['wall'].values())
                    self.reaches['reaches']['downstream']['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], width_in, self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches']['downstream']['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[('downstream', ts), 'elev'] = attributes['elevation'] + dz

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    if active_volume > 0:
                        self.reaches['reaches']['downstream']['gsd'] = update_bed_fractions(fractions, tot_qs_in, qs_kg,
                                                                                            active_volume, self.porosity, minimum_frac=0.003)
                    else:
                        self.reaches['reaches']['upstream']['gsd'] = self.boundary_conditions['reaches']['downstream']['gsd'] # fractions

            # if i in [10,50,100,200,500,1000,1500,10000,15000]:
            if trigger is True:
                self.serialize_timestep(f'../Outputs/{self.reach_name}_{i}.json')

        print(f'Total sediment in from upstream {tot_in}')
        logging.info(f'Total sediment in from upstream {tot_in}')

        self.save_df()


b_c = '../Inputs/boundary_conditions_woods2.json'
r_a = '../Inputs/reaches_woods2.json'
dis = '../Inputs/Woods_Q_1hr.csv'
t_i = 3600
whg_woods = [5.947, 0.115]
dhg_woods = [0.283, 0.402]
# vhg_woods = [0.687, 0.55]
vhg_woods = [1.3, 0.3]
whg_sc = [7.1, 0.25]  # check on this
dhg_sc = [0.272, 0.34]  # check on this
r_n = 'Woods'

inst = DepositEvolution(b_c, r_a, dis, t_i, whg_woods, dhg_woods, vhg_woods, r_n)
inst.simulation()
