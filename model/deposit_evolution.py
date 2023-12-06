import pandas as pd
import numpy as np
from math import atan, floor, pi
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
        out_sizes.append(0.5)
        out_sizes.append(-0.5)
        return out_sizes[0], out_sizes[1]

    elif 0.5 < fractions[1.0] < 0.84:
        out_sizes.append(0.5)
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

        if 2**-out_sizes[1] < 1.5 * 2**-out_sizes[0]:
            out_sizes.remove(out_sizes[1])
            val = 2**-out_sizes[0] * 1.5
            out_sizes.append(-np.log2(val))

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


def active_layer_depth(d50, d84, depth, slope, width=None, d50_wall=None):
    """
    Calculates an estimate for active layer thickness based on flow strength
    :param d50:
    :param d84:
    :param depth:
    :param slope:
    :param width:
    :param d50_wall:
    :return:
    """

    crit_shields_50 = max(0.043 * np.log(d84 / d50) - 0.0005, 0.04)  # make sure it's at least the 'average' val of 0.04
    shields_50 = (9810 * depth * max(0, slope)) / (1650 * 9.81 * d50)

    # from D. Vázquez‐Tarrío et al 2021 (Haschenburger 1999)
    if slope < 0.02:  # pool-riffle  (Montgomery and Buffington, pelucis and lamb 2017)
        active_lyr_mm = 0.8 * (shields_50 / crit_shields_50) ** 1.55 * (d50 * 1000)
    elif 0.02 <= slope < 0.04:  # plane bed
        active_lyr_mm = 0.75 * (shields_50 / crit_shields_50) ** 0.92 * (d50 * 1000)
    else:  # step-pool
        active_lyr_mm = 0.54 * (shields_50 / crit_shields_50) ** 3.16 * (d50 * 1000)

    # now for wall
    if width is not None:
        wall_frac = 1.9534 * (width / depth) ** -1.12
        shields_wall = wall_frac * shields_50
        active_wall_mm = 0.75 * (shields_wall / 0.045) ** 0.92 * (d50_wall * 1000)
    else:
        active_wall_mm = 0

    return active_lyr_mm / 1000, active_wall_mm / 1000


def count_fractions_to_volume(fractions):
    """
    Converts grain size fractions from counts to volumes
    :param fractions:
    :return:
    """

    # imagine counts are 1000 particles
    # counts = {phi: frac * 1000 for phi, frac in fractions.items()}
    # volumes = {phi: (4/3)*pi*(((2**-phi)/2)/1000)**3 * num for phi, num in counts.items()}
    # tot_vol = sum(volumes.values())
    # volume_fracs = {phi: vol / tot_vol for phi, vol in volumes.items()}
    tmp_vals = {phi: frac * (4/3)*pi*((2**-phi/1000)/2)**3 for phi, frac in fractions.items()}
    volume_fracs = {phi: val / sum(tmp_vals.values()) for phi, val in tmp_vals.items()}

    return volume_fracs


def subfractions(gsd_dict, dqs_dict):
    """take the non-zero subset of a qs dictionary and recalculate gsd_dict for those fractions"""
    tmp_dict = {key: val for key, val in gsd_dict.items() if dqs_dict[key] < 0}
    if sum(tmp_dict.values()) > 0:
        new_fracs = {}
        for key in gsd_dict.keys():
            if key in tmp_dict.keys():
                new_fracs[key] = tmp_dict[key] / sum(tmp_dict.values())
            else:
                new_fracs[key] = 0
    else:
        new_fracs = gsd_dict

    return new_fracs


class DepositEvolution:
    def __init__(self, bound_conds: str, reach_atts: str, discharge: str, time_interval: int, width_hydro_geom: List,
                 depth_hydro_geom: List, meas_slope: float, reach_name: str):
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

        with open(bound_conds) as f_bc:
            self.boundary_conditions = json.load(f_bc)
        with open(reach_atts) as f_ra:
            self.reaches = json.load(f_ra)
        self.w_geom = width_hydro_geom
        self.h_geom = depth_hydro_geom
        self.meas_slope = meas_slope
        self.reach_name = reach_name
        self.porosity = self.boundary_conditions['sediment_porosity']
        self.angle = self.boundary_conditions['ang_rep']

        self.q = pd.read_csv(discharge)
        if 'Q' not in self.q.columns:
            raise Exception('Column "Q" not found in discharge csv')
        if 'Datetime' not in self.q.columns:
            raise Exception('Column "Datetime" not found in discharge csv')

        self.time_interval = time_interval

        self.minimum_mass = {
            1: 1.73442094416937E-07,
            0: 1.38753675533549E-06,
            -1: 1.11002940426839E-05,
            -2: 8.88023523414715E-05,
            -2.5: 0.000251170982104,
            -3: 0.000710418818732,
            -3.5: 0.002009367856831,
            -4: 0.005683350549854,
            - 4.5: 0.016074942854649,
            -5: 0.045466804398833,
            -5.5: 0.12859954283719,
            -6: 0.363734435190667,
            -6.5: 1.02879634269752,
            -7: 2.90987548152534,
            -7.5: 8.23037074158015,
            -8: 23.2790038522027,
            -8.5: 65.8429659326412,
            - 9: 186.232030817622
        }

        self.max_mass = self.max_bed_mass()

        # set up log file
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Outputs')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'Outputs'))
        logfile = f'../Outputs/{self.reach_name}.log'
        logging.basicConfig(filename=logfile, format='%(levelname)s:%(message)s', level=logging.DEBUG)
        logging.info('Initializing model setup')

        # set up output table
        reach = [key for key in self.reaches['reaches'].keys()]
        reach.append('feeder')
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

        self.active_layer = {'upstream': ((2**-us_d50)/1000) * self.reaches['reaches']['upstream']['length'] * self.reaches['reaches']['upstream']['bankfull_width'],
                             'deposit_upstream': ((2**-us_d50)/1000) * self.reaches['reaches']['deposit_upstream']['length'] * self.reaches['reaches']['deposit_upstream']['bankfull_width'],
                             'deposit_downstream': ((2**-ds_d50)/1000) * self.reaches['reaches']['deposit_downstream']['length'] * self.reaches['reaches']['deposit_downstream']['bankfull_width'],
                             'downstream': ((2**-ds_d50)/1000) * self.reaches['reaches']['downstream']['length'] * self.reaches['reaches']['downstream']['bankfull_width']}

        # save initial elevations to track incision
        self.init_elevs = {key: self.reaches['reaches'][key]['elevation'] for key in self.reaches['reaches'].keys()}

        # set up deposit mass totals to keep track of
        dep_gsd_ct = {float(s): f for s, f in self.boundary_conditions['deposit_gsd'].items()}
        dep_gsd_vol = count_fractions_to_volume(dep_gsd_ct)
        self.df_mass = {}
        df_us_vol = (self.reaches['reaches']['deposit_upstream']['elevation'] -
                     self.boundary_conditions['reaches']['deposit_upstream']['pre_elev']) * \
                     self.reaches['reaches']['deposit_upstream']['bankfull_width'] * \
                     self.reaches['reaches']['deposit_upstream']['length']
        self.df_mass['deposit_upstream'] = {phi: frac * df_us_vol * (1 - self.porosity) * 2650 for phi, frac in dep_gsd_vol.items()}
        df_ds_vol = (self.reaches['reaches']['deposit_downstream']['elevation'] -
                     self.boundary_conditions['reaches']['deposit_downstream']['pre_elev']) * \
                    self.reaches['reaches']['deposit_downstream']['bankfull_width'] * \
                    self.reaches['reaches']['deposit_downstream']['length']
        self.df_mass['deposit_downstream'] = {phi: frac * df_ds_vol * (1 - self.porosity) * 2650 for phi, frac in dep_gsd_vol.items()}

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

    def incision_feedback(self, reach, incision_depth, active_vol_bed, active_vol_wall, gsd_bed_vol, gsd_wall_vol, ts):
        """
        Simulate bank failure resulting from incision
        :param reach:
        :param incision_depth:
        :param active_vol_bed:
        :param active_vol_wall:
        :param ts:
        :return:
        """

        # new bed gsd
        recruited = {}  # volume of each fraction recruited from bank
        for key, frac in gsd_wall_vol.items():
            min_vol = (self.minimum_mass[key] / (2650 * (1-self.porosity)))
            recruited[key] = floor((frac * active_vol_wall) / min_vol) * min_vol

        tot_vol = active_vol_bed + sum(recruited.values())

        bed_gsd_vol = \
            {key: ((gsd_bed_vol[key]*active_vol_bed) +
                  recruited[key]) / tot_vol for key in self.reaches['reaches'][reach]['gsd_bed'].keys()}
        self.reaches['reaches'][reach]['gsd_bed'] = self.mass_fracs_to_count(bed_gsd_vol, tot_vol)

        # new wall gsd
        remain = {key: (frac * active_vol_wall) - recruited[key] for key, frac in gsd_wall_vol.items()}

        # bound_gsd = {float(s): frac for s, frac in self.boundary_conditions['deposit_gsd'].items()}
        dep_gsd_mass = {phi: frac / sum(self.df_mass[reach].values()) for phi, frac in
                        self.df_mass[reach].items()}

        wall_gsd_vol = \
            {key: ((remain[key]) +
                  (dep_gsd_mass[key] * sum(recruited.values()))) /
                  active_vol_wall for key in gsd_wall_vol.keys()}
        self.reaches['reaches'][reach]['gsd_wall'] = self.mass_fracs_to_count(wall_gsd_vol, active_vol_wall)

        # adjust width
        dw = sum(recruited.values()) / ((self.reaches['reaches'][reach]['length'] * incision_depth) * 2)
        self.reaches['reaches'][reach]['width'] = self.reaches['reaches'][reach]['width'] + dw
        self.out_df.loc[(reach, ts), 'width'] = self.reaches['reaches'][reach]['width'] + dw

        # adjust elevation
        dz = sum(recruited.values()) / (self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['width'])
        self.reaches['reaches'][reach]['elevation'] = self.reaches['reaches'][reach]['elevation'] + dz
        self.out_df.loc[(reach, ts), 'elev'] = self.reaches['reaches'][reach]['elevation'] + dz

    def update_bed_fractions(self, reach, vol_fractions, qs_in, qs_out, active_volume, minimum_frac=None):
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

        if reach in ['deposit_upstream', 'deposit_downstream']:
            if sum(self.df_mass[reach].values()) == 0:
                dep_gsd_vol = {phi: 0 for phi in self.df_mass[reach].keys()}
            else:
                dep_gsd_vol = {phi: val / sum(self.df_mass[reach].values()) for phi, val in self.df_mass[reach].items()}
        else:
            dep_gsd_vol = None
        reach_gsd_ct = {float(s): f for s, f in self.boundary_conditions['reaches'][reach]['gsd'].items()}
        reach_gsd_vol = count_fractions_to_volume(reach_gsd_ct)

        # get change in mass for each fraction
        d_qs = {phi: qs_in[phi] - qs_out[phi] for phi in qs_in.keys()}

        # if there was no transport in or out, return the input gsd
        no_change = True
        for dqs in d_qs.values():
            if dqs != 0:
                no_change = False
                break

        if no_change is True:
            return self.mass_fracs_to_count(vol_fractions, active_volume), active_volume

        # find the new mass of each fraction and convert to new fractions
        new_mass = {phi: max(0, (vol_fractions[phi] * active_volume) * (1-self.porosity) * 2650 + d_qs[phi]) for phi in vol_fractions.keys()}

        # remove recruited fractions from deposit storage
        if reach in ('deposit_upstream', 'deposit_downstream'):
            for phi, val in d_qs.items():
                if val < 0:
                    if self.df_mass[reach][phi] > abs(val):
                        self.df_mass[reach][phi] += val
                    else:
                        self.df_mass[reach][phi] = 0.

        if sum(new_mass.values()) > 0:
            if sum(new_mass.values()) / (active_volume * (1 - self.porosity) * 2650) < 1:
                fraction_exposed = 1 - (sum(new_mass.values()) / (active_volume * (1 - self.porosity) * 2650))
                exposed_mass = (active_volume * (1 - self.porosity) * 2650) * fraction_exposed

                for phi, mass in new_mass.items():
                    if reach in ['upstream', 'downstream']:
                        # tmp_gsd = subfractions(reach_gsd_vol, d_qs)
                        # new_mass[phi] = mass + (tmp_gsd[phi] * exposed_mass)
                        new_mass[phi] = mass + reach_gsd_vol[phi] * exposed_mass
                    else:
                        # tmp_gsd = subfractions(dep_gsd_vol, d_qs)
                        # new_mass[phi] = mass + (tmp_gsd[phi] * exposed_mass)
                        new_mass[phi] = mass + dep_gsd_vol[phi] * exposed_mass

        # expose sediment below active layer if bed erodes - allows for armor breakup - using df mass tracking
        # if sum(d_qs.values()) < 0:
        # if sum(new_mass.values()) > 0:
        #     areas = {phi: (val / self.minimum_mass[phi]) * pi*((2 ** -phi / 1000)/2)**2 for phi, val in new_mass.items()}
        #     if sum(areas.values()) < bed_area:
        #         logging.info(f'remaining sediment less than bed area in {reach}')
        #         fraction_exposed = 1 - (sum(areas.values()) / bed_area)
        #         exposed_mass = sum(new_mass.values()) * fraction_exposed
        #
        #         for phi, mass in new_mass.items():
        #             if reach in ['upstream', 'downstream']:
        #                 new_mass[phi] = mass + (reach_gsd_vol[phi] * exposed_mass)
        #             else:
        #                 new_mass[phi] = mass + (dep_gsd_vol[phi] * exposed_mass)

            fractions_out_mass = {phi: new_mass[phi] / sum(new_mass.values()) for phi in new_mass.keys()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(new_mass.values()))

            rem_active_vol = sum(new_mass.values()) / ((1 - self.porosity) * 2650)

        else:
            if sum(dep_gsd_vol.values()) > 0:
                fractions_out_mass = dep_gsd_vol
                fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, active_volume * (1 - self.porosity) * 2650)
            else:
                fractions_out_ct = self.boundary_conditions['reaches'][reach]['gsd']

            rem_active_vol = 0

        # recalc = False
        # set_fracs = []
        # if reach in ['deposit_upstream', 'deposit_downstream']:
        #     for key in fractions_out_ct.keys():
        #         if fractions_out_ct[key] > self.max_fracs[reach][key]:
        #             recalc = True
        #             tot_mass_transfer = (fractions_out_ct[key] - self.max_fracs[reach][key]) * sum(new_mass.values())
        #             tmp_fracs = {phi: frac for phi, frac in dep_gsd.items() if phi > key and phi not in set_fracs}
        #             adj_fracs = {phi: frac / sum(tmp_fracs.values()) for phi, frac in tmp_fracs.items()}
        #             mass_transfer = {phi: frac * tot_mass_transfer for phi, frac in adj_fracs.items()}
        #             for phi, mass in new_mass.items():
        #                 if phi == key:
        #                     new_mass[phi] = new_mass[phi]- tot_mass_transfer
        #                     set_fracs.append(phi)
        #                 elif phi in mass_transfer.keys():
        #                     new_mass[phi] = new_mass[phi] + mass_transfer[phi]
        #
        # if recalc is True:
        #     fractions_out = {phi: new_mass[phi] / sum(new_mass.values()) for phi in new_mass.keys()}
        to_add = 0
        add_to = []
        for phi, frac in fractions_out_ct.items():
            if phi < -2 and frac > 0.2:
                to_add += frac - 0.2
                fractions_out_ct[phi] = 0.2
        for phi, frac in fractions_out_ct.items():
            if phi < -2 and frac < 0.2:
                add_to.append(phi)
        for phi in add_to:
            if fractions_out_ct[phi] + to_add / len(add_to) > 0.2:
                add_to.remove(phi)
        for phi, frac in fractions_out_ct.items():
            if phi in add_to:
                fractions_out_ct[phi] = frac + (to_add / len(add_to))
        # if fractions added are present in deposit, subtract from deposit?

        if minimum_frac:
            to_remove = 0
            remove_from = []
            for phi, frac in fractions_out_ct.items():
                if frac < minimum_frac:
                    to_remove += minimum_frac - frac
                    fractions_out_ct[phi] = minimum_frac
            for phi, frac in fractions_out_ct.items():
                if frac > to_remove:
                    remove_from.append(phi)
            # for phi, frac in fractions_out_ct.items():
            #     if frac > minimum_frac:
            #         if frac - to_remove < minimum_frac:
            #             remove_from.remove(phi)
            #         else:
            #             continue
            for phi, frac in fractions_out_ct.items():
                if phi in remove_from:
                    fractions_out_ct[phi] = frac - (to_remove / len(remove_from))

        for key, val in fractions_out_ct.items():
            if val < 0 or val is None:
                raise Exception('Fraction less than 0')

        if sum(fractions_out_ct.values()) > 1.1:
            raise Exception('Fractions sum to more than 1')
        if sum(fractions_out_ct.values()) < 0.95:
            raise Exception('Fractions sum to less than 1')

        return fractions_out_ct, rem_active_vol

    def update_wall_fractions(self, reach, vol_fractions, qs_out, active_volume_wall, minimum_frac=None):
        """
        Update the grain size fractions in the channel banks based on recruitment from banks by sediment transport
        :param reach:
        :param vol_fractions:
        :param qs_out:
        :param active_volume_wall:
        :param minimum_frac:
        :return:
        """

        # if there was no transport out, return the input gsd
        no_change = True
        for qs in qs_out.values():
            if qs != 0:
                no_change = False
                break

        if no_change is True:
            return self.mass_fracs_to_count(vol_fractions, active_volume_wall)

        # get existing mass of each fraction in active layer
        existing_mass = {phi: frac * active_volume_wall * (1 - self.porosity) * 2650 for phi, frac in vol_fractions.items()}

        # remove recruited fractions from deposit storage
        for phi, val in qs_out.items():
            if val > 0:
                if self.df_mass[reach][phi] > val:
                    self.df_mass[reach][phi] -= val
                else:
                    self.df_mass[reach][phi] = 0.

        # find the new mass of each fraction and convert to new fractions
        new_mass = {phi: max(0, existing_mass[phi] - qs_out[phi]) for phi in existing_mass.keys()}

        if sum(self.df_mass[reach].values()) > 0:
            dep_gsd_vol = {phi: val / sum(self.df_mass[reach].values()) for phi, val in self.df_mass[reach].items()}
        else:
            dep_gsd_vol = {phi: 0. for phi in self.df_mass.keys()}

        if sum(new_mass.values()) > 0:
            if sum(new_mass.values()) / (active_volume_wall * (1 - self.porosity) * 2650) < 1:
                d_qs = {key: -val for key, val in qs_out.items()}
                fraction_exposed = 1 - (sum(new_mass.values()) / (active_volume_wall * (1 - self.porosity) * 2650))
                exposed_mass = (active_volume_wall * (1 - self.porosity) * 2650) * fraction_exposed

                for phi, mass in new_mass.items():
                    tmp_gsd = subfractions(dep_gsd_vol, d_qs)
                    new_mass[phi] = mass + (tmp_gsd[phi] * exposed_mass)
                    # new_mass[phi] = mass + (dep_gsd_vol[phi] * exposed_mass)

            # if the area remaining is less than wall area recruit fractions from layers below

            # areas = {phi: (val / self.minimum_mass[phi]) * pi*((2 ** -phi / 1000)/2)**2 for phi, val in new_mass.items()}
            # if sum(areas.values()) < wall_area:
            #     fraction_exposed = 1 - (sum(areas.values()) / wall_area)
            #     exposed_mass = sum(new_mass.values()) * fraction_exposed
            #
            #     for phi, mass in new_mass.items():
            #         new_mass[phi] = mass + dep_gsd_vol[phi] * exposed_mass

            fractions_out_mass = {phi: new_mass[phi] / sum(new_mass.values()) for phi in new_mass.keys()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(new_mass.values()))

        else:
            if sum(dep_gsd_vol.values()) > 0:
                fractions_out_mass = dep_gsd_vol
                fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, active_volume_wall * (1 - self.porosity) * 2650)
            else:
                fractions_out_ct = self.boundary_conditions['reaches'][reach]['gsd']

        to_add = 0
        add_to = []
        for phi, frac in fractions_out_ct.items():
            if phi < -2 and frac > 0.3:
                to_add += frac - 0.3
                fractions_out_ct[phi] = 0.3
        for phi, frac in fractions_out_ct.items():
            if phi < -2 and frac < 0.3:
                add_to.append(phi)
        for phi in add_to:
            if fractions_out_ct[phi] + to_add / len(add_to) > 0.3:
                add_to.remove(phi)
        for phi, frac in fractions_out_ct.items():
            if phi in add_to:
                fractions_out_ct[phi] = frac + (to_add / len(add_to))

        if minimum_frac:
            to_remove = 0
            remove_from = []
            for phi, frac in fractions_out_ct.items():
                if frac < minimum_frac:
                    to_remove += minimum_frac - frac
                    fractions_out_ct[phi] = minimum_frac
            for phi, frac in fractions_out_ct.items():
                if frac > to_remove:
                    remove_from.append(phi)
            # for phi, frac in fractions_out_ct.items():
            #     if frac > minimum_frac:
            #         if frac - (to_remove / len(remove_from)) < minimum_frac:
            #             remove_from.remove(phi)
            #         else:
            #             continue
            for phi, frac in fractions_out_ct.items():
                if phi in remove_from:
                    fractions_out_ct[phi] = frac - (to_remove / len(remove_from))

        for key, val in fractions_out_ct.items():
            if val < 0:
                raise Exception('Fraction less than 0')
            # if val < 0.001 and key > -7:
            #     raise Exception('fraction less than 0.001')

        total = sum([frac for size, frac in fractions_out_ct.items()])
        if total > 1.1:
            raise Exception('Fractions sum to more than 1')
        if total < 0.95:
            raise Exception('Fractions sum to less than 1')

        return fractions_out_ct

    def max_bed_mass(self):
        """calculates the maximum bed fraction each size can possibly have based on volume
        delivered to channel from debris flow"""
        us_gsd = {float(s): f for s, f in self.boundary_conditions['reaches']['deposit_upstream']['gsd'].items()}
        ds_gsd = {float(s): f for s, f in self.boundary_conditions['reaches']['deposit_downstream']['gsd'].items()}
        dep_gsd = {float(s): f for s, f in self.boundary_conditions['deposit_gsd'].items()}
        dep_gsd_vol = count_fractions_to_volume(dep_gsd)

        us_d50, us_d84 = percentiles(us_gsd)
        us_d50, us_d84 = 2 ** -us_d50 / 1000, 2 ** -us_d84 / 1000
        upstream_area = self.reaches['reaches']['deposit_upstream']['bankfull_width'] * self.reaches['reaches']['deposit_upstream']['length']
        upstream_vols = {phi: (frac * (us_d50 * upstream_area)) for phi, frac in dep_gsd_vol.items()}

        ds_d50, ds_d84 = percentiles(ds_gsd)
        ds_d50, ds_d84 = 2 ** -ds_d50 / 1000, 2 ** -ds_d84 / 1000
        downstream_area = self.reaches['reaches']['deposit_downstream']['bankfull_width'] * \
                        self.reaches['reaches']['deposit_downstream']['length']
        downstream_vols = {phi: (frac * (ds_d50 * downstream_area)) for phi, frac in dep_gsd_vol.items()}

        max_vols = {
            'deposit_upstream': {phi: frac * 2650 * (1 - self.porosity) for phi, frac in upstream_vols.items()},
            'deposit_downstream': {phi: frac * 2650 * (1 - self.porosity) for phi, frac in downstream_vols.items()}
        }

        return max_vols

    def mass_fracs_to_count(self, mass_fractions, tot_mass):
        """
        Convert grain size fractions from mass values to counts
        :param mass_fractions:
        :param tot_mass:
        :return:
        """

        count = {phi: (frac * tot_mass) / self.minimum_mass[phi] for phi, frac in mass_fractions.items()}
        # count = {phi: ((frac * tot_mass) / ((1-self.porosity)*2650)) / ((4/3)*pi*((2**-phi/2)/1000)**3) for phi, frac in mass_fractions.items()}
        count_fractions = {phi: ct / sum(count.values()) for phi, ct in count.items()}

        return count_fractions

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
        kg_sm_recr = 0.0

        for i in tqdm(self.q.index):
            ts = self.q.loc[i, 'Datetime']
            self.reaches['timestep'] = ts
            q_in = self.q.loc[i, 'Q'] * self.boundary_conditions['flow_scale']
            slope = self.boundary_conditions['upstream_slope']
            width_in = self.w_geom[0] * q_in ** self.w_geom[1]
            if width_in > self.boundary_conditions['bankfull_width']:
                width_in = self.boundary_conditions['bankfull_width']
            depth_in = (self.h_geom[0] * q_in ** self.h_geom[1]) * (self.meas_slope / slope)**0.75
            fractions_in = {float(s): f for s, f in self.boundary_conditions['upstream_gsd'].items()}
            qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval)
            qs_kg = {key: val[1] for key, val in qs_fractions.items()}
            tot_in += sum(qs_kg.values())
            logging.info(f'{ts} Qs_in: {sum(qs_kg.values())}')
            transfer_vals['upstream']['Qs_in'] = qs_kg
            self.out_df.loc[('feeder', ts), 'yield'] = sum(qs_kg.values())

            for reach, attributes in self.reaches['reaches'].items():
                if reach == 'upstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd'].keys()) == 0:
                        self.reaches['reaches'][reach]['gsd'] = self.boundary_conditions['reaches'][reach]['gsd']
                        attributes['gsd'] = self.boundary_conditions['reaches'][reach]['gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals[reach]['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_upstream']['elevation']) / \
                            attributes['length']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    if slope > 0:
                        slope_corr = (self.meas_slope / slope)**0.75
                    else:
                        slope_corr = 1
                    depth_in = (self.h_geom[0] * q_in ** self.h_geom[1]) * min(slope_corr, 1.2)
                    self.out_df.loc[(reach, ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    volume_fractions = count_fractions_to_volume(fractions)
                    d50, d84 = percentiles(fractions)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000
                    # if d50 > 0.002:
                    # active_layer = 7968 * tau_star**2.61 * d50  # from Wilcock 1997
                    active_layer, _ = active_layer_depth(d50, d84, depth_in, slope)
                    if slope > 0 and active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022
                        logging.info(f'active layer: {active_layer} (recking max scour)')
                    if active_layer <= 0:  # because of - slope just make it positive to not break stuff
                        active_layer = d50
                    # else:
                    #     active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]
                    active_volume = attributes['length'] * width_in * active_layer

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {key: [0., 0.] for key in fractions.keys()}
                    else:
                        qs_fractions = transport(fractions, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: (frac * active_volume * (1 - self.porosity) * 2650) * 0.9 for key, frac in
                                    volume_fractions.items()}
                    for size, frac in qs_kg.items():
                        if frac > qs_in[size] + max_bed_recr[size]:
                            qs_kg[size] = qs_in[size] + max_bed_recr[size]

                    transfer_vals['upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_upstream']['Qs_in'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches'][reach]['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], width_in, self.porosity)  # porosity (0.21) from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz

                    # update bed fractions
                    # if active_volume > 0:
                    self.reaches['reaches'][reach]['gsd'], _ = self.update_bed_fractions(reach, volume_fractions, qs_in, qs_kg, active_volume,
                                                                                               minimum_frac=0.003)
                    # else:
                    #     self.reaches['reaches']['upstream']['gsd'] = self.boundary_conditions['reaches']['upstream']['gsd']  # fractions

                if reach == 'deposit_upstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd_bed'].keys()) == 0:
                        self.reaches['reaches'][reach]['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        self.reaches['reaches'][reach]['gsd_wall'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_wall'] = self.boundary_conditions['deposit_gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['deposit_upstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    slope = (attributes['elevation'] - self.reaches['reaches']['deposit_downstream']['elevation']) / \
                            attributes['length']
                    width_in = min(attributes['width'], self.w_geom[0] * q_in ** self.w_geom[1])
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    if slope > 0:
                        slope_corr = (self.meas_slope / slope) ** 0.75
                    else:
                        slope_corr = 1
                    depth_in = (self.h_geom[0] * q_in ** self.h_geom[1]) * min(slope_corr, 1.2)
                    self.out_df.loc[('deposit_upstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {'bed': {float(s): f for s, f in attributes['gsd_bed'].items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                    fractions_in_volume = {'bed': count_fractions_to_volume(fractions_in['bed']),
                                           'wall': count_fractions_to_volume(fractions_in['wall'])}
                    d50, d84 = percentiles(fractions_in['bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000
                    d50_wall, d84_wall = percentiles(fractions_in['wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000

                    # if d50 > 0.002:
                    # active_layer = 7968 * tau_star ** 2.61 * d50  # from Wilcock 1997
                    active_layer, active_layer_wall = active_layer_depth(d50, d84, depth_in, slope, width_in, d50_wall)
                    # if active_layer > 2 * d84:
                    #     active_layer = 2 * d84
                    lyr_ratio = min(1, active_layer_wall / active_layer)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022
                    if active_layer_wall > lyr_ratio * active_layer:
                        active_layer_wall = lyr_ratio * active_layer
                    # if active_layer_wall > d50_wall*0.5:
                    #     active_layer_wall = d50_wall*0.5
                    # else:
                    #     active_layer = 3 * d50
                    #     active_layer_wall = 2 * d50_wall
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * active_layer_wall * 2

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {'bed': {key: [0., 0.] for key in fractions_in['bed'].keys()},
                                        'wall': {key: [0., 0.] for key in fractions_in['wall'].keys()}}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                    qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                             'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}
                    ### testing for armor breakup
                    d50_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d50_phi <= -4:
                        if qs_kg['bed'][d50_phi] > 0:
                            print('breaking up armor')
                            dep_gsd_mass = {phi: val / sum(self.df_mass[reach].values()) for phi, val in self.df_mass[reach].items()}
                            dep_gsd_ct = self.mass_fracs_to_count(dep_gsd_mass, sum(self.df_mass[reach].values()))
                            fractions_in = {'bed': dep_gsd_ct,
                                            'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                            fractions_in_volume = {'bed': count_fractions_to_volume(fractions_in['bed']),
                                                   'wall': count_fractions_to_volume(fractions_in['wall'])}
                            qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                     twod=True, lwd_factor=attributes['lwd_factor'])
                            qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                                     'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: min(self.df_mass[reach][key], (frac * active_volume * (1 - self.porosity) * 2650) * 0.9) for key, frac in
                                    fractions_in_volume['bed'].items()}
                    max_wall_recr = {key: min(self.df_mass[reach][key], (frac * active_volume_wall * (1 - self.porosity) * 2650) * 0.9) for key, frac in
                                     fractions_in_volume['wall'].items()}

                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in[size] + max_bed_recr[size]:
                            qs_kg['bed'][size] = qs_in[size] + max_bed_recr[size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > max_wall_recr[size]:
                            qs_kg['wall'][size] = max_wall_recr[size]

                    kg_sm_recr += qs_kg['bed'][1.0] - qs_in[1.0]

                    transfer_vals['deposit_upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_downstream']['Qs_in'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg['bed'].values()) + \
                        sum(qs_kg['wall'].values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches'][reach]['Qs_out_bed'] = sum(qs_kg['bed'].values())
                    self.reaches['reaches'][reach]['Qs_out_wall'] = sum(qs_kg['wall'].values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_upstream']['Qs_in'],
                                                                transfer_vals['deposit_upstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches'][reach]['width'] = attributes['width'] + dw
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    # if sum(qs_in.values()) != 0 and sum(qs_kg['bed'].values()) != 0:
                    self.reaches['reaches'][reach]['gsd_bed'], active_volume = self.update_bed_fractions(reach,
                        fractions_in_volume['bed'], qs_in, qs_kg['bed'], active_volume, minimum_frac=0.001)
                    # else:
                    #     gsd_conv = {float(s): f for s, f in
                    #                 self.boundary_conditions['reaches']['deposit_downstream']['gsd'].items()}
                    #     self.reaches['reaches']['deposit_upstream']['gsd_bed'] = \
                    #         {key: (gsd_conv[key] + self.reaches['reaches']['deposit_upstream']['gsd_bed'][key]) /
                    #               2 for key in self.reaches['reaches']['deposit_upstream']['gsd_bed'].keys()}
                    if active_volume_wall > 0:
                        self.reaches['reaches'][reach]['gsd_wall'] = self.update_wall_fractions(reach,
                            fractions_in_volume['wall'], qs_kg['wall'], active_volume_wall)

                    # if the channel incises, expose new fractions in the walls
                    if dz < 0:
                        dep_gsd_mass = {phi: frac / sum(self.df_mass[reach].values()) for phi, frac in self.df_mass[reach].items()}
                        for size, frac in fractions_in_volume['wall'].items():
                            dep_frac = dep_gsd_mass[size]
                            new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                            new_frac = (frac * active_volume_wall + dep_frac * (attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                            fractions_in_volume['wall'][size] = new_frac
                            self.reaches['reaches'][reach]['gsd_wall'] = self.mass_fracs_to_count(fractions_in_volume['wall'], new_vol)

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs[reach] - attributes['elevation'] + dz
                    angle = atan(incision / ((attributes['width'] + dw) * 0.2)) # assumes bottom width is 60% top width
                    if angle > self.angle:
                        self.incision_feedback(reach, incision, active_volume, active_volume_wall, fractions_in_volume['bed'], fractions_in_volume['wall'], ts)
                        logging.info(f'bank slumping feedback for deposit upstream at timestep {i}')

                if reach == 'deposit_downstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd_bed'].keys()) == 0:
                        self.reaches['reaches'][reach]['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_bed'] = self.boundary_conditions['deposit_gsd']
                        self.reaches['reaches'][reach]['gsd_wall'] = self.boundary_conditions['deposit_gsd']
                        attributes['gsd_wall'] = self.boundary_conditions['deposit_gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['deposit_downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    slope = (attributes['elevation'] - self.reaches['reaches']['downstream']['elevation']) / \
                            attributes['length']
                    width_in = min(attributes['width'], self.w_geom[0] * q_in ** self.w_geom[1])
                    if width_in > self.boundary_conditions['bankfull_width']:
                        width_in = self.boundary_conditions['bankfull_width']
                    if slope > 0:
                        slope_corr = (self.meas_slope / slope) ** 0.75
                    else:
                        slope_corr = 1
                    depth_in = (self.h_geom[0] * q_in ** self.h_geom[1]) * min(slope_corr, 1.2)
                    self.out_df.loc[(reach, ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {'bed': {float(s): f for s, f in attributes['gsd_bed'].items()},
                                    'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                    fractions_in_volume = {'bed': count_fractions_to_volume(fractions_in['bed']),
                                           'wall': count_fractions_to_volume(fractions_in['wall'])}
                    d50, d84 = percentiles(fractions_in['bed'])
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000
                    d50_wall, d84_wall = percentiles(fractions_in['wall'])
                    d50_wall, d84_wall = (2 ** -d50_wall) / 1000, (2 ** -d84_wall) / 1000

                    # if d50 > 0.002:
                    active_layer, active_layer_wall = active_layer_depth(d50, d84, depth_in, slope, width_in, d50_wall)
                    lyr_ratio = min(1, active_layer_wall / active_layer)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022
                    if active_layer_wall > lyr_ratio * active_layer:
                        active_layer_wall = lyr_ratio * active_layer


                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * (active_layer_wall * 2)

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {'bed': {key: [0., 0.] for key in fractions_in['bed'].keys()},
                                        'wall': {key: [0., 0.] for key in fractions_in['wall'].keys()}}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                    qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                             'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # checking for armor breakup
                    d50_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d50_phi <= -4:
                        if qs_kg['bed'][d50_phi] > 0:
                            print('breaking up armor')
                            dep_gsd_mass = {phi: val / sum(self.df_mass[reach].values()) for phi, val in
                                            self.df_mass[reach].items()}
                            dep_gsd_ct = self.mass_fracs_to_count(dep_gsd_mass, sum(self.df_mass[reach].values()))
                            fractions_in = {'bed': dep_gsd_ct,
                                            'wall': {float(s): f for s, f in attributes['gsd_wall'].items()}}
                            fractions_in_volume = {'bed': count_fractions_to_volume(fractions_in['bed']),
                                                   'wall': count_fractions_to_volume(fractions_in['wall'])}
                            qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                     twod=True, lwd_factor=attributes['lwd_factor'])
                            qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                                     'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: (frac * active_volume * (1 - self.porosity) * 2650) * 0.9 for key, frac in
                                    fractions_in_volume['bed'].items()}
                    max_wall_recr = {key: (frac * active_volume_wall * (1 - self.porosity) * 2650) * 0.9 for key, frac
                                     in fractions_in_volume['wall'].items()}
                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]:
                            qs_kg['bed'][size] = qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > max_wall_recr[size]:
                            qs_kg['wall'][size] = max_wall_recr[size]

                    transfer_vals['deposit_downstream']['Qs_out'] = qs_kg
                    transfer_vals['downstream']['Qs_in'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg['bed'].values()) + \
                                                                           sum(qs_kg['wall'].values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in['bed'].values()) + \
                                                                             sum(qs_in['wall'].values())
                    self.reaches['reaches'][reach]['Qs_out_bed'] = sum(qs_kg['bed'].values())
                    self.reaches['reaches'][reach]['Qs_out_wall'] = sum(qs_kg['wall'].values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(transfer_vals['deposit_downstream']['Qs_in'],
                                                                transfer_vals['deposit_downstream']['Qs_out'])
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches'][reach]['width'] = attributes['width'] + dw
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    self.reaches['reaches'][reach]['gsd_bed'], active_volume = self.update_bed_fractions(reach,
                        fractions_in_volume['bed'], tot_qs_in, qs_kg['bed'], active_volume, minimum_frac=0.001)
                    # else:
                    #     gsd_conv = {float(s): f for s, f in self.boundary_conditions['reaches']['deposit_downstream']['gsd'].items()}
                    #     self.reaches['reaches']['deposit_downstream']['gsd_bed'] = \
                    #         {key: (gsd_conv[key] + self.reaches['reaches']['deposit_downstream']['gsd_bed'][key]) /
                    #               2 for key in self.reaches['reaches']['deposit_downstream']['gsd_bed'].keys()}
                    if active_volume_wall > 0:
                        self.reaches['reaches'][reach]['gsd_wall'] = self.update_wall_fractions(reach,
                            fractions_in_volume['wall'], qs_kg['wall'], active_volume_wall)

                    d50new, d84new = percentiles(self.reaches['reaches'][reach]['gsd_bed'])
                    d50new, d84new = (2 ** -d50new) / 1000, (2 ** -d84new) / 1000
                    # print(f'new d50: {d50new}')
                    # if d50new < 0.25*d50:
                    #     print('checking')

                    # if the channel incises, expose new fractions in the walls
                    if dz < 0:
                        dep_gsd_mass = {phi: frac / sum(self.df_mass[reach].values()) for phi, frac in
                                        self.df_mass[reach].items()}
                        for size, frac in fractions_in_volume['wall'].items():
                            dep_frac = dep_gsd_mass[size]
                            new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                            new_frac = (frac * active_volume_wall + dep_frac * (
                                        attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                            fractions_in_volume['wall'][size] = new_frac
                            self.reaches['reaches'][reach]['gsd_wall'] = self.mass_fracs_to_count(
                                fractions_in_volume['wall'], new_vol)

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs[reach] - attributes['elevation'] + dz
                    angle = atan(incision / ((attributes['width'] + dw) * 0.2))
                    if angle > self.angle:
                        vol_in = max(active_volume_wall, d50_wall * attributes['length'] * 2 * depth_in)
                        self.incision_feedback(reach, incision, active_volume, vol_in, fractions_in_volume['bed'], fractions_in_volume['wall'], ts)
                        logging.info(f'bank slumping feedback for deposit downstream at timestep {i}')

                if reach == 'downstream':
                    # if it's the first time step, copy over grain size distribution from boundary conditions
                    if len(attributes['gsd'].keys()) == 0:
                        self.reaches['reaches'][reach]['gsd'] = self.boundary_conditions['reaches'][reach]['gsd']
                        attributes['gsd'] = self.boundary_conditions['reaches'][reach]['gsd']

                    # calculate discharge, width, depth, and reach slope
                    qs_in = transfer_vals['downstream']['Qs_in']
                    q_in = self.q.loc[i, 'Q'] * attributes['flow_scale']
                    slope = (attributes['elevation'] - self.boundary_conditions['downstream_elev']) / attributes['length']
                    width_in = self.w_geom[0] * q_in ** self.w_geom[1]
                    if width_in > attributes['bankfull_width']:
                        width_in = attributes['bankfull_width']
                    if slope > 0:
                        slope_corr = (self.meas_slope / slope) ** 0.75
                    else:
                        slope_corr = 1
                    depth_in = (self.h_geom[0] * q_in ** self.h_geom[1]) * min(slope_corr, 1.2)
                    self.out_df.loc[('downstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions = {float(s): f for s, f in attributes['gsd'].items()}
                    fractions_volume = count_fractions_to_volume(fractions)
                    d50, d84 = percentiles(fractions)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000
                    # if d50 > 0.002:
                    active_layer, _ = active_layer_depth(d50, d84, depth_in, slope)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022
                    # else:
                    #     active_layer = 3 * d50
                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]
                    active_volume = attributes['length'] * width_in * active_layer

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {key: [0., 0.] for key in fractions.keys()}
                    else:
                        qs_fractions = transport(fractions, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check that qs for each fraction isn't more than what is available in active layer
                    max_bed_recr = {key: (frac * active_volume * (1 - self.porosity) * 2650) * 0.9 for key, frac in
                                    fractions_volume.items()}

                    for size, frac in qs_kg.items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]:
                            qs_kg[size] = qs_in['bed'][size] + qs_in['wall'][size] + max_bed_recr[size]

                    transfer_vals['downstream']['Qs_out'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in['bed'].values()) + \
                                                                     sum(qs_in['wall'].values())
                    self.reaches['reaches'][reach]['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], width_in, self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    # if active_volume > 0:
                    self.reaches['reaches']['downstream']['gsd'], _ = self.update_bed_fractions(reach, fractions_volume, tot_qs_in, qs_kg,
                                                                                            active_volume, minimum_frac=0.003)
                    # else:
                    #     self.reaches['reaches']['upstream']['gsd'] = self.boundary_conditions['reaches']['downstream']['gsd'] # fractions

            if i in [50,100,200,500,1000,1500,10000,15000, 20000, 30000]:
                self.serialize_timestep(f'../Outputs/{self.reach_name}_{i}.json')

        print(f'Total sediment in from upstream {tot_in}')
        print(f'Deposit mass: {self.df_mass}')
        logging.info(f'Total sediment in from upstream {tot_in}')
        logging.info(f'Deposit mass: {self.df_mass}')

        self.save_df()


b_c = '../Inputs/boundary_conditions_woods3.json'
r_a = '../Inputs/reaches_woods3.json'
dis = '../Inputs/Woods_Q_1hr.csv'
t_i = 3600
whg_woods = [4.688, 0.281]
dhg_woods = [0.282, 0.406]
# vhg_woods = [0.687, 0.55]
vhg_woods = [1.3, 0.3]
whg_sc = [6.2, 0.103]  # check on this
dhg_sc = [0.377, 0.39]  # check on this
vhg_sc = [0.95, 0.4]

whg_b = [9.14, 0.18]
dhg_b = [0.298, 0.143]
vhg_b = [0.367, 0.675]

r_n = 'Woods'

measurement_slope = 0.016

inst = DepositEvolution(b_c, r_a, dis, t_i, whg_b, dhg_b, measurement_slope, r_n)
inst.simulation()
