# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict
import csv
import json
import logging
import sys

from fastcache import lru_cache


__author__ = "Yaroslav Klyuyev <me@imposeren.org>"
__version__ = '0.0.1'


# probabilities for rolling 4d6 keep 3 largest
PRECALCED_4D6K3_PROBS = {
    3: 0.0008,
    4: 0.0031,
    5: 0.0077,
    6: 0.016,
    7: 0.029,
    8: 0.04,
    9: 0.07,
    10: 0.09,
    11: 0.11,
    12: 0.12,
    13: 0.13,
    14: 0.12,
    15: 0.10,
    16: 0.07,
    17: 0.04,
    18: 0.016,
}


# probabilities for rolling 2d10 keep largest 1
PRECALCED_2D10K1_PROBS = {
    1: 0.01,
    2: 0.03,
    3: 0.05,
    4: 0.07,
    5: 0.09,
    6: 0.11,
    7: 0.13,
    8: 0.15,
    9: 0.17,
    10: 0.19,
}

DEFAULT_POINTBUY_TABLE = {}

for v in range(8, 14):
    DEFAULT_POINTBUY_TABLE[v] = v - 8

DEFAULT_POINTBUY_TABLE[14] = 7  # +2
DEFAULT_POINTBUY_TABLE[15] = 9  # +2
DEFAULT_POINTBUY_TABLE[16] = 13  # +4
DEFAULT_POINTBUY_TABLE[17] = 17  # +4
DEFAULT_POINTBUY_TABLE[18] = 23  # +6
DEFAULT_POINTBUY_TABLE[7] = -1  # +6

if __name__ == '__main__':
    logger_name = 'stats_table'
    use_debug = True
else:
    logger_name = __name__
    use_debug = False

logger = logging.getLogger(logger_name)

if use_debug:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def lengths_to_coords(lengths_list, normalize=True):
    """convert list of ranges to list of (start, stop) tuples representing 1d coordinates ."""
    coords_list = []
    current_coord = 0

    for length in lengths_list:
        next_coord = current_coord + length
        coords_list.append((current_coord, next_coord))
        current_coord = next_coord

    if normalize:
        total_length = current_coord
        for (i, (start, stop)) in enumerate(coords_list):
            coords_list[i] = (start/total_length, stop/total_length)
    return coords_list


def get_overlap_length(cover_start, cover_end, target_start, target_end):
    overlap_start = max(cover_start, target_start)
    overlap_end = min(cover_end, target_end)
    if overlap_end > overlap_start:
        overlap = overlap_end - overlap_start
    else:
        overlap = 0

    return overlap


def get_dice_description(spec):
    num_dices, dice_size = spec
    if dice_size == 1:
        description = 'no roll required'
    elif dice_size in (2, 3, 4, 6, 8, 10, 12, 20, 100):
        description = '{}d{}'.format(num_dices, dice_size)
    elif spec == (1, 40):
        description = '(1d2-1)*20 + 1d20'
    else:
        raise ValueError('unknown dice spec `%s`' % (spec,))
    return description


def jsonize_tuple_keys(some_dict):
    if isinstance(some_dict, OrderedDict):
        tmp_dict = OrderedDict()
    else:
        tmp_dict = {}
    for key, val in some_dict.items():
        if isinstance(key, tuple):
            key = '.'.join([str(x) for x in key])
        elif not isinstance(key, str):
            key = str(key)
        if isinstance(val, dict):
            val = jsonize_tuple_keys(val)
        tmp_dict[key] = val
    return tmp_dict


class StatsTable(object):

    def __init__(
            self,
            min_stat_val=8,
            max_stat_val=16,
            min_point_buy_cost=27,
            max_point_buy_cost=27,
            max_probability_scale=120,
            num_stats=6,
            table_dices=((1, 100), (1, 40), (1, 20), (1, 12), (1, 10), (1, 8), (1, 6), (1, 4), (1, 2), (1, 1)),
            pointbuy_table=None,):
        """Initialize stats table calculator.

        :param int min_stat_val: minimum value for stat

        :param int max_stat_val: maximum value for stat

        :param int min_point_buy_cost: minimum pointbuy cost for stats set

        :param int max_point_buy_cost: maximum pointbuy cost for stats set

        :param float max_probability_scale:
            stat sets that have probability
            lower than max_probability/max_probability_scale
            will not be used.

        :param int num_stats: number of stats

        :param iterable table_dices:
            list of dices that are allowed to be used as table subselector, dices spec
            should be specified as 2 element tuple with ``(num_dices, dice_size)``

        :param int allowed_growth_size:
            how much calculated table may grow to match predefined sezes

        """
        self._set_of_stat_tuples = None

        if max_point_buy_cost < min_point_buy_cost:
            raise ValueError("'max_point_buy_cost' must be bigger than 'min_point_buy_cost'")

        if max_stat_val <= min_stat_val:
            raise ValueError("'max_stat_val' must be bigger than 'min_stat_val'")

        self.min_point_buy_cost = min_point_buy_cost
        self.max_point_buy_cost = max_point_buy_cost
        self.max_probability_scale = max_probability_scale

        self.min_stat_val = min_stat_val
        self.max_stat_val = max_stat_val
        self.stats_range = (min_stat_val, max_stat_val+1)

        self.num_stats = num_stats
        self.table_dices = table_dices
        self.__table_sizes = None
        self.__table_size_to_dices_mapping = None

        self.pointbuy_table = DEFAULT_POINTBUY_TABLE

        self._min_prob = 1
        self._max_prob = 0
        self.__max_prob_finalized = bool(not max_probability_scale)
        if self.__max_prob_finalized:
            self.__minimum_allowed_probability = 0
        else:
            self.__minimum_allowed_probability = None

    @property
    def set_of_stat_tuples(self):
        if self._set_of_stat_tuples is None:
            self.generate_stats_tuples()

        return self._set_of_stat_tuples

    @property
    def max_prob(self):
        if not self.__max_prob_finalized:
            self.generate_stats_tuples()
        return self._max_prob

    @property
    def min_prob(self):
        return self._min_prob

    @property
    def minimum_allowed_probability(self):
        if not self.__max_prob_finalized:
            raise RuntimeError("max probability is not yet calculated")
        if self.__minimum_allowed_probability is None:
            self.__minimum_allowed_probability = self.max_prob / self.max_probability_scale
        return self.__minimum_allowed_probability

    @property
    def table_dices(self):
        """Specification of table dices as passed on class init."""
        return self.__table_dices

    @table_dices.setter
    def table_dices(self, value):
        """Ensure that table_dices specs are sorted by size of a table."""
        self.__table_dices = list(value)
        self.__table_dices.sort(key=lambda x: x[0]*x[1], reverse=True)
        self.__table_dices = tuple(self.__table_dices)
        self.__table_sizes = None
        self.__table_size_to_dices_mapping = None

    @property
    def table_sizes(self):
        """Return sizes of tables that use ``self.table_dices``."""
        if self.__table_sizes is None:
            self.__table_sizes = []
            self.__table_size_to_dices_mapping = {}
            for spec in self.table_dices:
                size = spec[0] * spec[1]
                self.__table_sizes.append(size)
                if size in self.__table_size_to_dices_mapping:
                    raise ValueError("dice specs with matching sizes is not allowed")
                self.__table_size_to_dices_mapping[size] = spec
            self.__table_sizes = tuple(self.__table_sizes)
        return self.__table_sizes

    @property
    def talbe_size_to_dices_mapping(self):
        self.table_sizes
        return self.__table_size_to_dices_mapping

    @lru_cache(maxsize=128, typed=False)
    def calc_probability(self, stats_tuple):
        total_probability = 1
        for stat in stats_tuple:
            total_probability *= PRECALCED_4D6K3_PROBS[stat]
        return total_probability

    @lru_cache(maxsize=128, typed=False)
    def calc_pointbuy_cost(self, stats_tuple):
        return sum([self.pointbuy_table[stat] for stat in stats_tuple])

    def check_stats_correctness(self, stats_tuple):
        return (
            (self.min_point_buy_cost <= self.calc_pointbuy_cost(stats_tuple) <= self.max_point_buy_cost)
            and
            self.check_probability_correctness(stats_tuple, False)
        )

    def check_probability_correctness(self, stats_tuple, ensure_finalized=True):
        if ensure_finalized and not self.__max_prob_finalized:
            self.minimum_allowed_probability
        return (
            not self.__max_prob_finalized
            or
            self.calc_probability(stats_tuple) >= self.minimum_allowed_probability
        )

    def generate_stats_tuples(self):
        """Generate all possible stats sets."""
        self._set_of_stat_tuples = set()
        self._set_of_stat_tuples.add((self.min_stat_val, ) * self.num_stats)

        for stat_index in range(self.num_stats):
            extra_stats = []
            for added_stat in self._set_of_stat_tuples:
                for stat_val in range(*self.stats_range):
                    if stat_val <= added_stat[stat_index]:
                        continue
                    if stat_index > 0 and stat_val > added_stat[stat_index-1]:
                        break
                    new_stats_tuple = list(added_stat)
                    new_stats_tuple[stat_index] = stat_val
                    new_stats_tuple = tuple(new_stats_tuple)

                    # add even not correct values because correct values may
                    # be later built from them
                    extra_stats.append(new_stats_tuple)

                    if self.check_stats_correctness(new_stats_tuple):
                        prob = self.calc_probability(new_stats_tuple)
                        self._max_prob = max(prob, self._max_prob)

            self._set_of_stat_tuples.update(extra_stats)

        self.__max_prob_finalized = True

        self._set_of_stat_tuples = frozenset([
            stats_tuple for stats_tuple in self.set_of_stat_tuples
            if self.check_stats_correctness(stats_tuple)
        ])

    def get_cadidate_sizes(self, desired_size):
        """Get allowed table sizes that are closest to ``desired_size``."""
        if desired_size >= self.table_sizes[0]:
            return [self.table_sizes[0]]
        elif desired_size in (self.table_sizes + (0, )):
            return [desired_size]
        else:
            smaller_size = 0
            bigger_size = self.table_sizes[0]
            for tested_size in self.table_sizes[1:]:
                if desired_size > tested_size:
                    smaller_size = tested_size
                    break
                else:
                    bigger_size = tested_size

        return [smaller_size, bigger_size]

    def get_fitting_variants_generator(self, prefered_table_sizes):
        fitting_candidate = list(prefered_table_sizes)

        for i, prefered_table_size in enumerate(prefered_table_sizes):
            size_candidates = self.get_cadidate_sizes(prefered_table_size)

            if len(size_candidates) == 1:
                fitting_candidate[i] = size_candidates[0]
                if i+1 == len(prefered_table_sizes):
                    yield fitting_candidate
            else:
                for size_candidate in size_candidates:
                    partial_sizes = list(prefered_table_sizes[i+1:])
                    len_partial_sizes = len(partial_sizes)
                    if len_partial_sizes:
                        # Next size must be tuned if current size differs from desired
                        delta = prefered_table_size - size_candidate
                        partial_sizes[0] += delta
                        if partial_sizes[0] < 0:
                            partial_sizes[:] = [0] * len_partial_sizes
                            yield fitting_candidate[:i] + [size_candidate] + partial_sizes
                        else:
                            for sub_data in self.get_fitting_variants_generator(partial_sizes):
                                yield fitting_candidate[:i] + [size_candidate] + sub_data
                    else:
                        yield fitting_candidate[:i] + [size_candidate]
                break

    def get_fit_quality(self, fit_results, prefered_results):
        logger.debug('processing fit candidate: %s', fit_results)
        item_checks = []

        fit_coords = lengths_to_coords(fit_results)
        prefered_coords = lengths_to_coords(prefered_results, normalize=True)

        total_midpoint_offset = 0
        total_abs_midpoint_offset = 0
        processed = 0

        for (fitted_segment, prefered_segment) in zip(fit_coords, prefered_coords):
            processed += 1
            fitted_start, fitted_end = fitted_segment
            prefered_start, prefered_end = prefered_segment

            fitted_len = fitted_end - fitted_start
            prefered_len = prefered_end - prefered_start

            fitted_midpoint = (fitted_start + fitted_end) / 2
            prefered_midpoint = (prefered_start + prefered_end) / 2

            midpoint_offset = fitted_midpoint - prefered_midpoint
            abs_midpoint_offset = abs(midpoint_offset)
            total_abs_midpoint_offset += abs_midpoint_offset
            total_midpoint_offset += midpoint_offset

            overlap_len = get_overlap_length(fitted_start, fitted_end, prefered_start, prefered_end)

            rolling_avg_midpoint_offset = total_midpoint_offset/processed

            offseted_overlap_len = get_overlap_length(
                fitted_start-rolling_avg_midpoint_offset, fitted_end-rolling_avg_midpoint_offset,
                prefered_start, prefered_end
            )
            if prefered_len == 0:
                import pdb; pdb.set_trace()

            item_checks.append({
                'items': {
                    'fitted': {
                        'start': fitted_start,
                        'end': fitted_end,
                        'len': fitted_len,
                    },
                    'prefered': {
                        'start': prefered_start,
                        'end': prefered_end,
                        'len': prefered_len,
                    },
                },
                'length_rel_error': abs(fitted_len - prefered_len) / prefered_len,

                'midpoint_offset': midpoint_offset,
                'rolling_avg_midpoint_offset': rolling_avg_midpoint_offset,

                'overlap_len': overlap_len,
                'overlap_ratio': overlap_len / prefered_len,

                'offseted_overlap_len': offseted_overlap_len,
                'offseted_overlap_ratio': offseted_overlap_len / prefered_len,
            })

        quality = 0

        for (i, datum) in enumerate(item_checks):
            fitted_segment = datum['items']['fitted']
            prefered_segment = datum['items']['prefered']

            prefered_len = prefered_segment['len']
            fitted_len = fitted_segment['len']

            gracious_overlap_quality = max(datum['overlap_ratio'], datum['offseted_overlap_ratio'])

            offset_quality = 1 - datum['midpoint_offset']
            len_quality = 1 - datum['length_rel_error']

            if fitted_len:

                # offset is considered most important, then comes length and overlapping
                item_quality = (4*offset_quality + 3*len_quality + 1*gracious_overlap_quality)/(4+3+1)
                len_norm_coeff = 1/(prefered_len)**0.25
                # add some weight for shorter segments:
                item_quality *= len_norm_coeff
            else:
                item_quality = 0

            logger.debug('qualities of segment #{0}: offset={1}, len={2}, overlap={3}, len_norm_coeff={4}, total={5}'.format(i, offset_quality, len_quality, gracious_overlap_quality, len_norm_coeff, item_quality))

            quality += item_quality

        logger.debug('total quality of %s: %s', fit_results, quality)
        return quality

    def get_strange_tables(self, prefered_table_sizes=None):
        """Return data for "strange" stat choosing tables.

        Up to 10 "tables" are generated.
        Table number is chosen by rolling 2d10 and keeping largest one.
        Each table may specify additional dices to be rolled

        """

        # 1. Roll to get number of table: (2d10 keep largest 1).
        table_number_rolls = range(1, 11)

        # Now I perform something similar to projection that makes
        # ranges for 2 probabilities the same.
        # This is not mathematically correct but gives satisfying results
        table_number_rolls_max_prob = PRECALCED_2D10K1_PROBS[10]

        proj_coeff = self.max_prob / table_number_rolls_max_prob
        proj_coeff *= 1.01

        projected_table_number_rolls_probs = {}

        for table_number in table_number_rolls:
            roll_prob = PRECALCED_2D10K1_PROBS[table_number]
            projected_prob = roll_prob * proj_coeff
            projected_table_number_rolls_probs[table_number] = projected_prob

        stats_tuples_with_prob = [
            (stats_tuple, self.calc_probability(stats_tuple))
            for stats_tuple in self.set_of_stat_tuples
        ]

        stats_tuples_with_prob.sort(key=lambda x: x[1])

        table_number_rolls = list(table_number_rolls)
        table_number_rolls.sort(key=lambda x: projected_table_number_rolls_probs[x])

        if prefered_table_sizes is None:
            prefered_table_sizes = []

        # index of first stats set that is not yet processed
        stats_processing_start_index = sum(prefered_table_sizes)

        # index of first table to be processed:
        tables_processing_start_index = len(prefered_table_sizes)

        for table_number in table_number_rolls[tables_processing_start_index:]:
            prefered_table_size = 0

            current_table_projected_prob = projected_table_number_rolls_probs.get(table_number, 1)
            next_table_projected_prob = projected_table_number_rolls_probs.get(table_number+1, 1)

            if next_table_projected_prob == 1:
                limiting_prob = 1
            else:
                limiting_prob = (current_table_projected_prob + next_table_projected_prob) / 2

            while stats_tuples_with_prob[stats_processing_start_index][1] < limiting_prob:
                prefered_table_size += 1
                stats_processing_start_index += 1

                if (stats_processing_start_index+1) >= len(stats_tuples_with_prob):
                    break
            prefered_table_sizes.append(prefered_table_size)

            if (stats_processing_start_index+1) >= len(stats_tuples_with_prob):
                break

        prefered_table_sizes = prefered_table_sizes + [0] * (10-len(prefered_table_sizes))
        logger.info("prefered sizes of tables: %s", prefered_table_sizes)

        total_quality = 0
        best_quality = 0
        for fitted_sizes in self.get_fitting_variants_generator(prefered_table_sizes):
            quality = self.get_fit_quality(fitted_sizes, prefered_table_sizes)
            total_quality += quality
            best_quality = max(quality, best_quality)
            if quality == best_quality:
                best_fitting_sizes = fitted_sizes

        logger.info("best fitting table sizes: %s", best_fitting_sizes)

        table_data = OrderedDict()
        for i, size in enumerate(best_fitting_sizes):
            table_number = i + 1
            dice_to_stats_mapping = OrderedDict()
            probability = PRECALCED_2D10K1_PROBS[table_number]
            table_data[table_number] = {
                'subrolls': get_dice_description(self.talbe_size_to_dices_mapping[size]),
                'stats': dice_to_stats_mapping,
                'probability': probability,
            }

            for j in range(1, size+1):
                if size == 40:
                    if j <= 20:
                        subkey = (1, j)
                    else:
                        subkey = (2, j-20)
                else:
                    subkey = (j, )
                dice_key = (table_number, ) + subkey

                stats, prob = stats_tuples_with_prob.pop(0)
                stats_as_string = ' '.join([str(x) for x in stats])

                dice_to_stats_mapping[dice_key] = {
                    'values': stats,
                    'str': stats_as_string,
                    'pointbuy_cost': self.calc_pointbuy_cost(stats),
                }
        return table_data


if __name__ == '__main__':
    st = StatsTable()

    results1 = jsonize_tuple_keys(st.get_strange_tables())

    # print(json.dumps(results1, indent=2))

    st = StatsTable(
        min_stat_val=7,
        max_stat_val=17,
        min_point_buy_cost=26,
        max_point_buy_cost=28,
        max_probability_scale=1000,
    )

    results2 = jsonize_tuple_keys(st.get_strange_tables())

    # print(json.dumps(results2, indent=2))

    for results, filename in ((results1, 'table_avg.csv'), (results2, 'table_bigger.csv')):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for table_num, table_data in results.items():
                writer.writerow([table_num, 'Chance: %.2f%%' % (table_data['probability']*100), 'Subroll: %s' % table_data['subrolls']])
                writer.writerow(['Rolls', 'Stats', 'PB cost'])
                for dice_key, stat_data in table_data['stats'].items():
                    writer.writerow([dice_key, stat_data['str'], stat_data['pointbuy_cost']])
