import pandas as pd
import numpy as np
from math import atan, floor, pi, ceil
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
    if dw < 0:
        print('checking')

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
        active_wall_mm = 0.8 * (shields_wall / 0.045) ** 1.55 * (d50_wall * 1000)
    else:
        active_wall_mm = 0

    return active_lyr_mm / 1000, active_wall_mm / 1000


def count_fractions_to_volume(fractions):
    """
    Converts grain size fractions from counts to volumes
    :param fractions:
    :return:
    """

    # find the number of particles needed based on smallest non-zero fraction
    num = 1 / (min([val for val in fractions.values() if val > 0]))
    counts = {phi: frac * num for phi, frac in fractions.items()}
    volumes = {phi: ((4/3)*pi*(((2**-phi)/2)/1000)**3) * ct for phi, ct in counts.items()}
    volume_fracs = {phi: vol / sum(volumes.values()) for phi, vol in volumes.items()}
    # tmp_vals = {phi: frac * (4/3)*pi*((2**-phi/1000)/2)**3 for phi, frac in fractions.items()}
    # volume_fracs = {phi: val / sum(tmp_vals.values()) for phi, val in tmp_vals.items()}

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

        # self.max_mass = self.max_bed_mass()

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
        us_thickness = (2 * (2**-us_d50) / 1000)
        us_minelev = self.reaches['reaches']['upstream']['elevation'] - us_thickness
        df_us_minelev = self.boundary_conditions['reaches']['deposit_upstream']['pre_elev'] - us_thickness
        ds_gsd = {float(s): f for s, f in self.boundary_conditions['reaches']['downstream']['gsd'].items()}
        ds_d50, ds_d84 = percentiles(ds_gsd)
        ds_thickness = (2 * (2 ** -ds_d50) / 1000)
        ds_minelev = self.reaches['reaches']['downstream']['elevation'] - ds_thickness
        df_ds_minelev = self.boundary_conditions['reaches']['deposit_downstream']['pre_elev'] - ds_thickness

        self.boundary_conditions['min_elev'] = {'upstream': us_minelev, 'deposit_upstream': df_us_minelev,
                                                 'deposit_downstream': df_ds_minelev, 'downstream': ds_minelev}

        # save initial elevations to track incision
        self.init_elevs = {key: self.reaches['reaches'][key]['elevation'] for key in self.reaches['reaches'].keys()}

        # set up deposit mass totals to keep track of
        dep_gsd_ct = {float(s): f for s, f in self.boundary_conditions['deposit_gsd'].items()}
        dep_gsd_vol = count_fractions_to_volume(dep_gsd_ct)
        us_gsd_ct = {float(s): f for s, f in self.boundary_conditions['reaches']['upstream']['gsd'].items()}
        us_gsd_vol = count_fractions_to_volume(us_gsd_ct)
        ds_gsd_ct = {float(s): f for s, f in self.boundary_conditions['reaches']['downstream']['gsd'].items()}
        ds_gsd_vol = count_fractions_to_volume(ds_gsd_ct)
        self.sed_mass = {}
        us_vol = (2 ** -us_d50 / 1000) * self.reaches['reaches']['upstream']['length'] * \
                 self.reaches['reaches']['upstream']['bankfull_width']
        self.sed_mass['upstream'] = {phi: frac * us_vol * (1 - self.porosity) * 2650 for phi, frac in
                                     us_gsd_vol.items()}
        ds_vol = (2 ** -ds_d50 / 1000) * self.reaches['reaches']['downstream']['length'] * \
                 self.reaches['reaches']['downstream'][
                     'bankfull_width']
        self.sed_mass['downstream'] = {phi: frac * ds_vol * (1 - self.porosity) * 2650 for phi, frac in
                                       ds_gsd_vol.items()}
        df_us_vol = (self.reaches['reaches']['deposit_upstream']['elevation'] -
                     self.boundary_conditions['reaches']['deposit_upstream']['pre_elev']) * \
                     self.reaches['reaches']['deposit_upstream']['bankfull_width'] * \
                     self.reaches['reaches']['deposit_upstream']['length']
        self.sed_mass['deposit_upstream'] = {phi: frac * df_us_vol * (1 - self.porosity) * 2650 for phi, frac in dep_gsd_vol.items()}
        df_ds_vol = (self.reaches['reaches']['deposit_downstream']['elevation'] -
                     self.boundary_conditions['reaches']['deposit_downstream']['pre_elev']) * \
                    self.reaches['reaches']['deposit_downstream']['bankfull_width'] * \
                    self.reaches['reaches']['deposit_downstream']['length']
        self.sed_mass['deposit_downstream'] = {phi: frac * df_ds_vol * (1 - self.porosity) * 2650 for phi, frac in dep_gsd_vol.items()}
        self.dus_mass_st = sum(self.sed_mass['deposit_upstream'].values())
        self.dds_mass_st = sum(self.sed_mass['deposit_downstream'].values())

        # keep an initial copy for mass/volume comparison
        self.init_sed_mass = self.sed_mass.copy()
        self.init_volumes = {
            'upstream': us_vol,
            'deposit_upstream': df_us_vol,
            'deposit_downstream': df_ds_vol,
            'downstream': ds_vol
        }

        # set up a dict for surface layer tracking
        self.surface_sed = {'upstream': {}, 'deposit_upstream': {}, 'deposit_downstream': {}, 'downstream': {}}
        self.bank_sed = {'deposit_upstream': {}, 'deposit_downstream': {}}

        self.us_mass_st = sum(self.sed_mass['upstream'].values())
        self.ds_mass_st = sum(self.sed_mass['downstream'].values())

    def serialize_timestep(self, outfile):
        """
        Save a .json file of a given time step
        :param outfile:
        :return:
        """

        with open(outfile, 'w') as dst:
            self.reaches['sed_masses'] = self.sed_mass
            json.dump(self.reaches, dst, indent=4)

    def save_df(self):
        """
        Save the output data frame to a csv file
        :return:
        """
        logging.info('Saving output csv')
        self.out_df.to_csv(f'../Outputs/{self.reach_name}_out.csv')

    def incision_feedback(self, reach, incision_depth, active_volume_wall, ts):
        """
        Simulate bank failure resulting from incision
        :param reach:
        :param incision_depth:
        :param active_vol_bed:
        :param active_vol_wall:
        :param ts:
        :return:
        """

        # init wall mass
        wall_mass = sum(self.bank_sed[reach].values())

        # transfer bank storage to bed
        transfer_vol = 0
        for phi, val in self.bank_sed[reach].items():
            self.surface_sed[reach][phi] += floor(val / self.minimum_mass[phi]) * self.minimum_mass[phi]
            self.bank_sed[reach][phi] -= floor(val / self.minimum_mass[phi]) * self.minimum_mass[phi]
            transfer_vol += (floor(val / self.minimum_mass[phi]) * self.minimum_mass[phi]) / ((1 - self.porosity) * 2650)
        # update bed gsd
        bed_gsd_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in self.surface_sed[reach].items()}
        self.reaches['reaches'][reach]['gsd_bed'] = self.mass_fracs_to_count(bed_gsd_mass, sum(self.surface_sed[reach].values()))

        # new wall gsd
        min_mass = 1 * active_volume_wall * (1 - self.porosity) * 2650
        repl_mass = max(min_mass - sum(self.bank_sed[reach].values()), 0.25 * sum(self.bank_sed[reach].values()))  # made up 0.25
        store_gsd_mass = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in self.sed_mass[reach].items()}
        for phi, frac in store_gsd_mass.items():
            self.bank_sed[reach][phi] += frac * repl_mass
            self.sed_mass[reach][phi] -= frac * repl_mass
        wall_gsd_mass = {phi: val / sum(self.bank_sed[reach].values()) for phi, val in self.bank_sed[reach].items()}
        self.reaches['reaches'][reach]['gsd_wall'] = self.mass_fracs_to_count(wall_gsd_mass, sum(self.bank_sed[reach].values()))

        # adjust width
        dw = transfer_vol / ((self.reaches['reaches'][reach]['length'] * incision_depth) * 2)
        if dw < 0:
            print('checking')
        self.reaches['reaches'][reach]['width'] = self.reaches['reaches'][reach]['width'] + dw
        self.out_df.loc[(reach, ts), 'width'] = self.reaches['reaches'][reach]['width'] + dw

        # adjust elevation
        dz = transfer_vol / (self.reaches['reaches'][reach]['length'] * self.reaches['reaches'][reach]['width'])
        self.reaches['reaches'][reach]['elevation'] = self.reaches['reaches'][reach]['elevation'] + dz
        self.out_df.loc[(reach, ts), 'elev'] = self.reaches['reaches'][reach]['elevation'] + dz

    def update_surface_sed(self, reach, min_mass=None, recruit=True, active_mass=None, bank=False):
        # if surface sediment is less than minimum mass recruit from subsurface
        if bank is False:
            if recruit is True:
                if min_mass is None:
                    raise Exception('Must have min_mass arg if recruit is True')
                recruit_mass = min_mass - sum(self.surface_sed[reach].values())
                sub_gsd_mass = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in self.sed_mass[reach].items()}

                for phi, val in sub_gsd_mass.items():
                    self.surface_sed[reach][phi] += min(val * recruit_mass, self.sed_mass[reach][phi])
                    self.sed_mass[reach][phi] -= min(val * recruit_mass, self.sed_mass[reach][phi])

        if bank is False:
            if recruit is False:
                if active_mass is None:
                    raise Exception('Must have active_mass arg if recruit is False')
                store_mass = sum(self.surface_sed[reach].values()) - active_mass
                surf_gsd_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in self.surface_sed[reach].items()}

                for phi, val in surf_gsd_mass.items():
                    self.sed_mass[reach][phi] += min(val * store_mass, self.surface_sed[reach][phi])
                    self.surface_sed[reach][phi] -= min(val * store_mass, self.surface_sed[reach][phi])

        if bank is True:
            if recruit is True:
                if min_mass is None:
                    raise Exception('Must have min_mass arg if recruit is True')
            recruit_mass = min_mass - sum(self.bank_sed[reach].values())
            sub_gsd_mass = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in self.sed_mass[reach].items()}

            for phi, val in sub_gsd_mass.items():
                self.bank_sed[reach][phi] += min(val * recruit_mass, self.sed_mass[reach][phi])
                self.sed_mass[reach][phi] -= min(val * recruit_mass, self.sed_mass[reach][phi])

    def update_bed_fractions(self, reach, qs_in, qs_out, active_volume, ct_fractions, minimum_frac=False, conserve=True):

        # if not minimum_frac:
        #     minimum_frac = 0

        # for fixing rounding problems between d_qs and change in storage
        start_store = sum(self.surface_sed[reach].values())

        # get change in mass for each fraction
        d_qs = {phi: (qs_in[phi] - qs_out[phi]) for phi in qs_in.keys()}

        # if there was no transport in or out, return the input gsd
        no_change = True
        for dqs in d_qs.values():
            if dqs != 0:
                no_change = False
                break

        if no_change is True:
            return ct_fractions, active_volume

        # find the new mass of each fraction and convert to new fractions
        # new_mass = {phi: max(0, (self.surface_sed[reach][phi] + d_qs[phi])) for phi in vol_fractions.keys()}

        # remove recruited fractions from surface storage
        for phi, val in d_qs.items():
            if val < 0:
                if ct_fractions[phi] != 0.:
                    if self.surface_sed[reach][phi] > abs(val):
                        self.surface_sed[reach][phi] += val
                    else:
                        self.surface_sed[reach][phi] = 0.
            else:
                self.surface_sed[reach][phi] += val

        min_mass = 1 * active_volume * (1 - self.porosity) * 2650 # DEFINE SOME MINIMUM MASS - made up 0.5 active vol
        if sum(self.surface_sed[reach].values()) > 0:
            if sum(self.surface_sed[reach].values()) < min_mass and sum(self.sed_mass[reach].values()) > 0:
                self.update_surface_sed(reach, min_mass)  # get the mass dif and introduce sed from subsurface based on fractions * mass dif

            elif sum(self.surface_sed[reach].values()) > (active_volume * (1-self.porosity) * 2650):  # if there was net deposition move some of that into subsurface storage? maybe use same function as above
                self.update_surface_sed(reach, min_mass, recruit=False, active_mass=(active_volume * (1-self.porosity) * 2650))

            fractions_out_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in self.surface_sed[reach].items()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(self.surface_sed[reach].values()))

            rem_active_vol = sum(self.surface_sed[reach].values()) / ((1 - self.porosity) * 2650)

        else:  # if the whole surface layer is used up
            store_gsd_mass = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in self.sed_mass[reach].items()}
            for phi, val in store_gsd_mass.items():
                self.surface_sed[reach][phi] += min(val * active_volume * (1-self.porosity) * 2650, self.sed_mass[reach][phi])
                self.sed_mass[reach][phi] -= min(val * active_volume * (1-self.porosity) * 2650, self.sed_mass[reach][phi])
            fractions_out_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in self.surface_sed[reach].items()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(fractions_out_mass.values()))

            rem_active_vol = 0

        # to_add = 0
        # add_to = []
        # for phi, frac in fractions_out_ct.items():
        #     if -7 < phi < -2 and frac > 0.15:
        #         to_add += frac - 0.15
        #         fractions_out_ct[phi] = 0.15
        #     if -8 <= phi <= -7 and frac > 0.1:
        #         to_add += frac - 0.1
        #         fractions_out_ct[phi] = 0.1
        #     if phi < -8 and frac > 0.05:
        #         to_add += frac - 0.05
        #         fractions_out_ct[phi] = 0.05
        # for phi, frac in fractions_out_ct.items():
        #     if -7 < phi < -2 and frac < 0.15:
        #         add_to.append(phi)
        #     if -8 <= phi <= -7 and frac < 0.1:
        #         add_to.append(phi)
        #     if phi < -8 and frac < 0.05:
        #         add_to.append(phi)
        # for phi in add_to:
        #     if -7 < phi < -2:
        #         if fractions_out_ct[phi] + to_add / len(add_to) > 0.15:
        #             add_to.remove(phi)
        #     if -7 <= phi <= -8:
        #         if fractions_out_ct[phi] + to_add / len(add_to) > 0.1:
        #             add_to.remove(phi)
        #     if phi < -8:
        #         if fractions_out_ct[phi] + to_add / len(add_to) > 0.05:
        #             add_to.remove(phi)
        # for phi, frac in fractions_out_ct.items():
        #     if phi in add_to:
        #         fractions_out_ct[phi] = frac + (to_add / len(add_to))
        # if fractions added are present in deposit, subtract from deposit?

        if minimum_frac is True:
            mass_ratio = (rem_active_vol * (1 - self.porosity) * 2650) / sum(self.surface_sed[reach].values())
            if mass_ratio >= 1:  # default 1 but maybe test other vals.
                min_mass = min([val for val in self.surface_sed[reach].values() if val != 0])
                for phi, val in self.surface_sed[reach].items():
                    if phi > -5 and val < min_mass:
                        if conserve is True:
                            self.surface_sed[reach][phi] += min(min_mass, self.sed_mass[reach][phi])
                            self.sed_mass[reach][phi] -= min(min_mass, self.sed_mass[reach][phi])
                        else:
                            self.surface_sed[reach][phi] += min_mass

            fractions_out_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in
                                  self.surface_sed[reach].items()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(self.surface_sed[reach].values()))

            rem_active_vol = sum(self.surface_sed[reach].values()) / ((1 - self.porosity) * 2650)

            # to_remove = 0
            # remove_from = []
            # for phi, frac in fractions_out_ct.items():
            #     if frac < minimum_frac:
            #         to_remove += minimum_frac - frac
            #         fractions_out_ct[phi] = minimum_frac
            # for phi, frac in fractions_out_ct.items():
            #     if frac > to_remove:
            #         remove_from.append(phi)
            # for phi, frac in fractions_out_ct.items():
            #     if phi in remove_from:
            #         fractions_out_ct[phi] = frac - (to_remove / len(remove_from))

        for key, val in fractions_out_ct.items():
            if val < 0 or val is None:
                raise Exception('Fraction less than 0')

        if sum(fractions_out_ct.values()) > 1.1 and sum(fractions_out_ct.values()) != 0:
            raise Exception('Fractions sum to more than 1')
        if sum(fractions_out_ct.values()) < 0.95 and sum(fractions_out_ct.values()) != 0:
            raise Exception('Fractions sum to less than 1')
        if min(self.surface_sed[reach].values()) < 0:
            raise Exception('negative surface values')

        # testing a fix for discrepancies in mass
        # tot_d_qs = sum(d_qs.values())
        # nonzs = [key for key, val in d_qs.items() if val != 0]
        # if reach in ('deposit_upstream', 'deposit_downstream'):
        #     d_store = sum(self.df_mass[reach].values()) - start_store
        #     dif = tot_d_qs - d_store
        #     for key in self.df_mass[reach].keys():
        #         if key in nonzs:
        #             if self.df_mass[reach][key] + dif / len(nonzs) > 0:
        #                 self.df_mass[reach][key] += dif / len(nonzs)
        #             else:
        #                 self.df_mass[reach][key] = 0
        # else:
        #     d_store = sum(self.sed_mass[reach].values()) - start_store
        #     dif = tot_d_qs - d_store
        #     for key in self.sed_mass[reach].keys():
        #         if key in nonzs:
        #             if self.sed_mass[reach][key] + dif / len(nonzs) > 0:
        #                 self.sed_mass[reach][key] += dif / len(nonzs)
        #             else:
        #                 self.sed_mass[reach][key] = 0

        return fractions_out_ct, rem_active_vol

    def update_wall_fractions(self, reach, qs_out, active_volume_wall, ct_fractions=None, minimum_frac=None):
        """
        Update the grain size fractions in the channel banks based on recruitment from banks by sediment transport
        :param reach:
        :param vol_fractions:
        :param qs_out:
        :param active_volume_wall:
        :param minimum_frac:
        :return:
        """

        if not minimum_frac:
            minimum_frac = 0.

        # if there was no transport out, return the input gsd
        no_change = True
        for qs in qs_out.values():
            if qs != 0:
                no_change = False
                break

        if no_change is True:
            return ct_fractions

        # get existing mass of each fraction in active layer
        # existing_mass = {phi: frac * active_volume_wall * (1 - self.porosity) * 2650 for phi, frac in vol_fractions.items()}

        # remove recruited fractions from deposit storage
        for phi, val in qs_out.items():
            if val > 0:
                if ct_fractions[phi] != 0.:
                    if self.bank_sed[reach][phi] > (1 - (minimum_frac / ct_fractions[phi])) * val:
                        self.bank_sed[reach][phi] -= (1 - (minimum_frac / ct_fractions[phi])) * val
                    else:
                        self.bank_sed[reach][phi] = 0.

        #debug
        # nonzs = [key for key, val in qs_out.items() if val != 0]
        # tot_qs = sum(qs_out.values())
        # store_dif = sum(self.df_mass[reach].values()) - start_store
        # dif = tot_qs + store_dif
        # for key in self.df_mass[reach].keys():
        #     if key in nonzs:
        #         if self.df_mass[reach][key] - dif > 0:
        #             self.df_mass[reach][key] -= dif / len(nonzs)
        #         else:
        #             self.df_mass[reach][key] = 0.

        # find the new mass of each fraction and convert to new fractions

        min_mass = 1 * active_volume_wall * (1 - self.porosity) * 2650
        if sum(self.bank_sed[reach].values()) > 0:
            if sum(self.bank_sed[reach].values()) < min_mass:
                self.update_surface_sed(reach, min_mass=min_mass, recruit=True, bank=True)

            fractions_out_mass = {phi: val / sum(self.bank_sed[reach].values()) for phi, val in
                                  self.bank_sed[reach].items()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(self.bank_sed[reach].values()))

        else:
            store_gsd_mass = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in
                              self.sed_mass[reach].items()}
            for phi, val in store_gsd_mass.items():
                self.bank_sed[reach][phi] += min(val * active_volume_wall * (1 - self.porosity) * 2650,
                                                    self.sed_mass[reach][phi])
                self.sed_mass[reach][phi] -= min(val * active_volume_wall * (1 - self.porosity) * 2650,
                                                 self.sed_mass[reach][phi])
            fractions_out_mass = {phi: val / sum(self.bank_sed[reach].values()) for phi, val in
                                  self.bank_sed[reach].items()}
            fractions_out_ct = self.mass_fracs_to_count(fractions_out_mass, sum(fractions_out_mass.values()))

        # to_add = 0
        # add_to = []
        # for phi, frac in fractions_out_ct.items():
        #     if phi < -2 and frac > 0.3:
        #         to_add += frac - 0.3
        #         fractions_out_ct[phi] = 0.3
        # for phi, frac in fractions_out_ct.items():
        #     if phi < -2 and frac < 0.3:
        #         add_to.append(phi)
        # for phi in add_to:
        #     if fractions_out_ct[phi] + to_add / len(add_to) > 0.3:
        #         add_to.remove(phi)
        # for phi, frac in fractions_out_ct.items():
        #     if phi in add_to:
        #         fractions_out_ct[phi] = frac + (to_add / len(add_to))

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

        # count = {phi: (frac * tot_mass) / self.minimum_mass[phi] for phi, frac in mass_fractions.items()}
        count = {phi: ((frac * tot_mass) / ((1-self.porosity)*2650)) / ((4/3)*pi*((2**-phi/2)/1000)**3) for phi, frac in mass_fractions.items()}
        count_fractions = {phi: ct / sum(count.values()) for phi, ct in count.items()}

        return count_fractions

    def vol_fracs_to_count(self, vol_fractions, tot_vol):

        count = {phi: (frac * tot_vol) / ((4/3)*pi*((2**-phi/2)/1000)**3) for phi, frac in vol_fractions.items()}
        count_fractions = {phi: ct / sum(count.values()) for phi, ct in count.items()}

        return count_fractions

    def initial_surface_mass(self, reach, vol, gsd_vol, bank=False):
        # set up initial surface layer
        self.surface_sed[reach] = {phi: frac * vol * (1 - self.porosity) * 2650 for phi, frac in gsd_vol.items()}
        # remove surface layer masses from storage masses
        for phi, mass in self.surface_sed[reach].items():
            self.sed_mass[reach][phi] -= min(mass, 0.9*self.sed_mass[reach][phi])

        if bank is True:
            self.bank_sed[reach] = {phi: frac * vol * (1 - self.porosity) * 2650 for phi, frac in gsd_vol.items()}
            for phi, mass in self.bank_sed[reach].items():
                self.sed_mass[reach][phi] -= mass

    def armor_breakup(self, reach, active_vol):
        store_gsd_vol = {phi: val / sum(self.sed_mass[reach].values()) for phi, val in self.sed_mass[reach].items()}
        dif_vol = max(active_vol - (sum(self.surface_sed[reach].values()) / ((1 - self.porosity) * 2650)), 0.5 * active_vol)  # made up 0.25
        for phi, val in store_gsd_vol.items():
            self.surface_sed[reach][phi] += min(val * dif_vol, self.sed_mass[reach][phi])
            self.sed_mass[reach][phi] -= min(val * dif_vol, self.sed_mass[reach][phi])

        gsd_mass = {phi: val / sum(self.surface_sed[reach].values()) for phi, val in self.surface_sed[reach].items()}
        gsd_ct = self.mass_fracs_to_count(gsd_mass, sum(self.surface_sed[reach].values()))

        return gsd_ct, gsd_mass

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

        tot_in = 0.0
        dus_in = 0.0
        dus_out = 0.0
        dds_in = 0.0
        dds_out = 0.0

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
            # logging.info(f'{ts} Qs_in: {sum(qs_kg.values())}')
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
                    chan_area = attributes['length'] * width_in
                    self.out_df.loc[(reach, ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {float(s): f for s, f in attributes['gsd'].items()}
                    volume_fractions = count_fractions_to_volume(fractions_in)

                    d50, d84 = percentiles(fractions_in)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000

                    active_layer, _ = active_layer_depth(d50, d84, depth_in, abs(slope))
                    if slope > 0 and active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope

                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]
                    active_volume = attributes['length'] * width_in * active_layer

                    # if it's the first time step or the entire surface layer was depleted in previous timestep
                    if len(self.surface_sed[reach].keys()) == 0:
                        self.initial_surface_mass(reach, active_volume, volume_fractions)

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {key: [0., 0.] for key in fractions_in.keys()}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    d84_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d84_phi <= -5 and qs_kg[d84_phi] > 0 and sum(self.sed_mass[reach].values()) > 0:
                        logging.info(f'armor breakup reach: {reach} timestep: {ts} flow: {q_in}')
                        fractions_in, volume_fractions = self.armor_breakup(reach, active_volume)
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                        qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check that qs for each fraction isn't more than what is available in active layer
                    for size, frac in qs_kg.items():
                        if frac > qs_in[size] + self.surface_sed[reach][size]:
                            qs_kg[size] = qs_in[size] + self.surface_sed[reach][size]

                    transfer_vals['upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_upstream']['Qs_in'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches'][reach]['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], attributes['bankfull_width'], self.porosity)  # porosity (0.21) from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz

                    # update bed fractions
                    self.reaches['reaches'][reach]['gsd'], _ = self.update_bed_fractions(reach, qs_in, qs_kg, active_volume, fractions_in)

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
                    chan_area = attributes['length'] * width_in
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

                    active_layer, active_layer_wall = active_layer_depth(d50, d84, depth_in, slope, width_in, d50_wall)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022
                    lyr_ratio = min(1, active_layer_wall / active_layer)
                    if active_layer_wall > lyr_ratio * active_layer:
                        active_layer_wall = lyr_ratio * active_layer

                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * active_layer_wall * 2

                    if len(self.surface_sed[reach].keys()) == 0:
                        self.initial_surface_mass(reach, active_volume, fractions_in_volume['bed'])
                        self.initial_surface_mass(reach, active_volume_wall, fractions_in_volume['wall'], bank=True)

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

                    # armor breakup
                    d84_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d84_phi <= -5 and qs_kg['bed'][d84_phi] > 0 and sum(self.sed_mass[reach].values()) > 0:
                        logging.info(f'armor breakup reach: {reach} timestep: {ts} flow: {q_in}')
                        fractions_in['bed'], fractions_in_volume['bed'] = self.armor_breakup(reach, active_volume)
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                        qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                                 'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    # check that qs for each fraction isn't more than what is available in active layer
                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in[size] + self.surface_sed[reach][size]:
                            qs_kg['bed'][size] = qs_in[size] + self.surface_sed[reach][size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > self.bank_sed[reach][size]:
                            qs_kg['wall'][size] = self.bank_sed[reach][size]

                    if sum(qs_kg['bed'].values()) + sum(qs_kg['wall'].values()) < 0:
                        print('wtf now')
                    transfer_vals['deposit_upstream']['Qs_out'] = qs_kg
                    transfer_vals['deposit_downstream']['Qs_in'] = qs_kg
                    dus_in += sum(transfer_vals['deposit_upstream']['Qs_in'].values())
                    dus_out += sum(transfer_vals['deposit_upstream']['Qs_out']['bed'].values()) + sum(
                        transfer_vals['deposit_upstream']['Qs_out']['wall'].values())

                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg['bed'].values()) + \
                        sum(qs_kg['wall'].values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in.values())
                    self.reaches['reaches'][reach]['Qs_out_bed'] = sum(qs_kg['bed'].values())
                    self.reaches['reaches'][reach]['Qs_out_wall'] = sum(qs_kg['wall'].values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], attributes['width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches'][reach]['width'] = attributes['width'] + dw
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    self.reaches['reaches'][reach]['gsd_bed'], active_volume = self.update_bed_fractions(reach,
                        qs_in, qs_kg['bed'], active_volume, fractions_in['bed'], minimum_frac=True)

                    self.reaches['reaches'][reach]['gsd_wall'] = self.update_wall_fractions(reach,
                        qs_kg['wall'], active_volume_wall, fractions_in['wall'])

                    # if the channel incises, expose new fractions in the walls
                    # if dz < 0:
                    #     if sum(self.df_mass[reach].values()) > active_volume_wall * (1-self.porosity) * 2650:
                    #         dep_gsd_mass = {phi: frac / sum(self.df_mass[reach].values()) for phi, frac in self.df_mass[reach].items()}
                    #         for size, frac in fractions_in_volume['wall'].items():
                    #             dep_frac = dep_gsd_mass[size]
                    #             new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                    #             new_frac = (frac * active_volume_wall + dep_frac * (attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                    #             fractions_in_volume['wall'][size] = new_frac
                    #         self.reaches['reaches'][reach]['gsd_wall'] = self.vol_fracs_to_count(fractions_in_volume['wall'], active_volume_wall)

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs[reach] - attributes['elevation'] + dz
                    angle = atan(incision / ((attributes['width'] + dw) * 0.2)) # assumes bottom width is 60% top width
                    if angle > self.angle and attributes['width'] + dw < self.reaches['reaches'][reach]['bankfull_width']:
                        self.incision_feedback(reach, incision, active_volume_wall, ts)
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
                    chan_area = attributes['length'] * width_in
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

                    active_layer, active_layer_wall = active_layer_depth(d50, d84, depth_in, slope, width_in, d50_wall)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope
                    lyr_ratio = min(1, active_layer_wall / active_layer)
                    if active_layer_wall > lyr_ratio * active_layer:
                        active_layer_wall = lyr_ratio * active_layer

                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]

                    active_volume = attributes['length'] * width_in * active_layer
                    active_volume_wall = attributes['length'] * depth_in * (active_layer_wall * 2)

                    if len(self.surface_sed[reach].keys()) == 0:
                        self.initial_surface_mass(reach, active_volume, fractions_in_volume['bed'])
                        self.initial_surface_mass(reach, active_volume_wall, fractions_in_volume['wall'], bank=True)

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
                    d84_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d84_phi <= -5 and qs_kg['bed'][d84_phi] > 0 and sum(self.sed_mass[reach].values()) > 0:
                        logging.info(f'armor breakup reach: {reach} timestep: {ts} flow: {q_in}')
                        fractions_in['bed'], fractions_in_volume['bed'] = self.armor_breakup(reach, active_volume)
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 twod=True, lwd_factor=attributes['lwd_factor'])
                        qs_kg = {'bed': {key: val[1] for key, val in qs_fractions['bed'].items()},
                                 'wall': {key: val[1] for key, val in qs_fractions['wall'].items()}}

                    for size, frac in qs_kg['bed'].items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + self.surface_sed[reach][size]:
                            qs_kg['bed'][size] = qs_in['bed'][size] + qs_in['wall'][size] + self.surface_sed[reach][size]
                    for size, frac in qs_kg['wall'].items():
                        if frac > self.bank_sed[reach][size]:
                            qs_kg['wall'][size] = self.bank_sed[reach][size]

                    transfer_vals['deposit_downstream']['Qs_out'] = qs_kg
                    transfer_vals['downstream']['Qs_in'] = qs_kg
                    dds_in += sum(transfer_vals['deposit_downstream']['Qs_in']['bed'].values()) + sum(transfer_vals['deposit_downstream']['Qs_in']['wall'].values())
                    dds_out += sum(transfer_vals['deposit_downstream']['Qs_out']['bed'].values()) + sum(transfer_vals['deposit_downstream']['Qs_out']['wall'].values())
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
                    # if dz > 0.05:
                    #     print('checking')
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    dw = delta_width(d_qs_wall, attributes['length'], depth_in, self.porosity)
                    self.reaches['reaches'][reach]['width'] = attributes['width'] + dw
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'width'] = attributes['width'] + dw

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    self.reaches['reaches'][reach]['gsd_bed'], active_volume = self.update_bed_fractions(reach,
                        tot_qs_in, qs_kg['bed'], active_volume, fractions_in['bed'], minimum_frac=True)

                    self.reaches['reaches'][reach]['gsd_wall'] = self.update_wall_fractions(reach,
                        qs_kg['wall'], active_volume_wall, fractions_in['wall'])

                    # if the channel incises, expose new fractions in the walls
                    # if dz < 0:
                    #     if sum(self.df_mass[reach].values()) > active_volume_wall * (1-self.porosity) * 2650:
                    #         dep_gsd_mass = {phi: frac / sum(self.df_mass[reach].values()) for phi, frac in
                    #                         self.df_mass[reach].items()}
                    #         for size, frac in fractions_in_volume['wall'].items():
                    #             dep_frac = dep_gsd_mass[size]
                    #             new_vol = active_volume_wall + (attributes['length'] * abs(dz) * active_layer * 2)
                    #             new_frac = (frac * active_volume_wall + dep_frac * (
                    #                         attributes['length'] * abs(dz) * active_layer * 2)) / new_vol
                    #             fractions_in_volume['wall'][size] = new_frac
                    #         self.reaches['reaches'][reach]['gsd_wall'] = self.vol_fracs_to_count(
                    #                 fractions_in_volume['wall'], active_volume_wall)

                    # if incision passes angle of repose trigger bank sloughing feedback
                    incision = self.init_elevs[reach] - attributes['elevation'] + dz
                    angle = atan(incision / ((attributes['width'] + dw) * 0.2))
                    if angle > self.angle and attributes['width'] + dw < self.reaches['reaches'][reach]['bankfull_width']:
                        self.incision_feedback(reach, incision, active_volume_wall, ts)
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
                    chan_area = attributes['length'] * width_in
                    self.out_df.loc[('downstream', ts), 'slope'] = slope

                    # find the D50 and D84 and estimate the active layer/volume
                    fractions_in = {float(s): f for s, f in attributes['gsd'].items()}
                    volume_fractions = count_fractions_to_volume(fractions_in)
                    d50, d84 = percentiles(fractions_in)
                    d50, d84 = (2 ** -d50) / 1000, (2 ** -d84) / 1000
                    self.out_df.loc[(reach, ts), 'D50'] = d50 * 1000

                    active_layer, _ = active_layer_depth(d50, d84, depth_in, slope)
                    if active_layer > 1.4 * width_in * slope:
                        active_layer = 1.4 * width_in * slope  # maximum scour depth from Recking et al 2022

                    if attributes['elevation'] - active_layer < self.boundary_conditions['min_elev'][reach]:
                        active_layer = attributes['elevation'] - self.boundary_conditions['min_elev'][reach]
                    active_volume = attributes['length'] * width_in * active_layer

                    if len(self.surface_sed[reach].keys()) == 0:
                        self.initial_surface_mass(reach, active_volume, volume_fractions)

                    self.reaches['reaches'][reach]['D50'] = d50
                    self.reaches['reaches'][reach]['D84'] = d84

                    # calculate transport
                    if slope <= 0:
                        qs_fractions = {key: [0., 0.] for key in fractions_in.keys()}
                    else:
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                    qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    # check for armor breakup
                    d84_phi = round(-np.log2(d50 * 1000) * 2) / 2
                    if d84_phi <= -5 and qs_kg[d84_phi] > 0 and sum(self.sed_mass[reach].values()) > 0:
                        logging.info(f'armor breakup reach: {reach} timestep: {ts} flow: {q_in}')
                        fractions_in, volume_fractions = self.armor_breakup(reach, active_volume)
                        qs_fractions = transport(fractions_in, slope, q_in, depth_in, width_in, self.time_interval,
                                                 lwd_factor=attributes['lwd_factor'])
                        qs_kg = {key: val[1] for key, val in qs_fractions.items()}

                    for size, frac in qs_kg.items():
                        if frac > qs_in['bed'][size] + qs_in['wall'][size] + self.surface_sed[reach][size]:
                            qs_kg[size] = qs_in['bed'][size] + qs_in['wall'][size] + self.surface_sed[reach][size]

                    if sum(qs_kg.values()) < 0:
                        print('wtf now')
                    transfer_vals['downstream']['Qs_out'] = qs_kg
                    self.out_df.loc[(reach, ts), 'yield'] = sum(qs_kg.values())
                    self.reaches['reaches'][reach]['Qs_in'] = sum(qs_in['bed'].values()) + \
                                                                     sum(qs_in['wall'].values())
                    self.reaches['reaches'][reach]['Qs_out'] = sum(qs_kg.values())

                    # geomorphic change
                    d_qs_bed, d_qs_wall = sediment_calculations(qs_in, qs_kg)
                    dz = exner(d_qs_bed, attributes['length'], attributes['bankfull_width'], self.porosity)  # porosity from Wu and Wang 2006 Czuba 2018
                    self.reaches['reaches'][reach]['elevation'] = attributes['elevation'] + dz
                    self.out_df.loc[(reach, ts), 'elev'] = attributes['elevation'] + dz

                    # update fractions
                    tot_qs_in = {key: qs_in['bed'][key] + qs_in['wall'][key] for key in qs_in['bed'].keys()}
                    self.reaches['reaches']['downstream']['gsd'], _ = self.update_bed_fractions(reach, tot_qs_in, qs_kg,
                                                                                            active_volume, fractions_in)


            for reach in self.sed_mass.values():
                for val in reach.values():
                    if val < 0:
                        print('stop and check')
            for reach in self.surface_sed.values():
                for val in reach.values():
                    if val < 0:
                        print('stop and check')

            if i in [50,100,200,500,1000,1500,10000,15000, 20000, 30000]:
                self.serialize_timestep(f'../Outputs/{self.reach_name}_{i}.json')

        dus_mass_e = sum(self.sed_mass['deposit_upstream'].values())
        dds_mass_e = sum(self.sed_mass['deposit_downstream'].values())
        us_mass_e = sum(self.sed_mass['upstream'].values())
        ds_mass_e = sum(self.sed_mass['downstream'].values())

        print(f'dep downstream in: {dds_in}')
        print(f'dep downstream yield: {dds_out}')
        print(f'difference in upstream mass: {us_mass_e - self.us_mass_st}')
        print(f'difference in deposit upstream mass: {dus_mass_e - self.dus_mass_st}')
        print(f'difference in deposit downstream mass: {dds_mass_e - self.dds_mass_st}')
        print(f'difference in downstream mass: {ds_mass_e - self.ds_mass_st}')
        print(f'Total sediment in from upstream {tot_in}')
        print(f'Deposit mass: {self.sed_mass}')
        logging.info(f'Total sediment in from upstream {tot_in}')
        logging.info(f'Subsurface storage: {self.sed_mass}')
        logging.info(f'Surface storage: {self.surface_sed}')

        self.save_df()


b_c = '../Inputs/boundary_conditions_blodgett.json'
r_a = '../Inputs/reaches_blodgett.json'
dis = '../Inputs/Blodgett_Q_2010_2020_1hr.csv'
t_i = 3600
whg_woods = [4.688, 0.281]
dhg_woods = [0.282, 0.406]
# vhg_woods = [0.687, 0.55]
vhg_woods = [1.3, 0.3]
whg_sc = [5.072, 0.377]  # check on this
dhg_sc = [0.23, 0.343]  # check on this
vhg_sc = [0.95, 0.4]

whg_b = [9.14, 0.18]
dhg_b = [0.298, 0.143]
vhg_b = [0.367, 0.675]

whg_rl = [5.213, 0.324]
dhg_rl = [0.227, 0.386]

r_n = 'Blodgett'

measurement_slope = 0.011

inst = DepositEvolution(b_c, r_a, dis, t_i, whg_sc, dhg_sc, measurement_slope, r_n)
inst.simulation()
