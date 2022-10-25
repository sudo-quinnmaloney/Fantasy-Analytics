from espn_api.basketball import League
from pandas import DataFrame, concat
from scipy.stats import norm
# from math import comb, exp, pi, sqrt
import sys
from time import time

LEAGUE_ID = 1825567028
YEAR = 2023
ESPN_S2 = 'AEBaPMLGilU%2FQHusMQxEx5A%2F8Ij4XnywMYpfWmcYB8iMKe7Uer6pWSy90mw4CSKRAtQJALoj5IMpKsYhSzxKvkoZu6A774RReJCO5WGELbuADXizQszW0hhcdWc7%2FMKOM%2BuPtDJH1NHXTdtXZz%2BIlLzpP1wD8SrRK2ElPaOSI1BS4HPdEs70WZ6s5Dvf3R0tg8arwxUYX9DCWVPWfJ47LouLcCfUXq%2Fh%2BXUExGxikwr2dRgheiW8kp6das5JdsouZ0wVlq4p%2FBLOlAG0zRObmDuN'
SWID = '{F7BA750E-1099-42C9-B323-763C553D8F8F}'

INTERMEDIATE_PARAMS = ['FGA', 'FGM', 'FG%', 'FTA', 'FTM', 'FT%', '3PTM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']
SCORING_PARAMS = ['FG%', 'FT%', '3PTM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS']

use_last_7 = True

# TODO: use class attributes for repetitively used shtuff
class FantasyAnalytics(League):
    def __init__(self, league_id, year, espn_s2, swid, home):
        League.__init__(self, league_id, year, espn_s2, swid)
        self.home_team = home
        self.comp_type = '2023_last_7'
        self.comp_totals, self.stat_stds = self.gen_totals_matrix()

    def map_team_names(self, teams: list) -> list:
        # cast single inputs as arrays
        if not isinstance(teams, list):
            teams = [teams]

        # make map
        team_dict = {}
        for t in self.teams:
            team_dict[t.team_name] = t.team_id

        # map names
        team_ids = []
        for team in teams:
            try:
                if isinstance(team, str):
                    team_ids.append(team_dict[team])
                elif isinstance(team, int) and team in team_dict.keys():
                    team_ids.append(int)
                else:
                    raise KeyError(team)
            except KeyError as ke:
                raise KeyError(f'Invalid team: {ke}')
        return team_ids, team_dict

    def get_teams(self, teams: list = []):
        # check input type
        team_ids, team_dict = self.map_team_names(teams)

        # return team_name/team_id map if none specified
        if not team_ids:
            return team_dict

        # get team data
        res = {}
        for t, i in team_dict.items():
            if i in team_ids:
                res[t] = self.get_team_data(i)
        return res

    def get_team(self, team: str or int):
        return list(self.get_teams(team).values())[0]

    def get_lineups(self, period: int = 1) -> dict:
        lineups = {}
        box = self.box_scores(matchup_period=period, scoring_period=period)
        for matchup in box:
            lineups[matchup.home_team.team_name] = matchup.home_lineup
            lineups[matchup.away_team.team_name] = matchup.away_lineup
        return lineups

    def get_rosters(self, period: int = 1) -> dict:
        rosters = {}
        for team in self.teams:
            rosters[team.team_name] = team.roster
        return rosters

    def get_stats_projected(self, player):
        try:
            projected_avgs = player.stats['2023_projected']['avg']
        except KeyError as ke:
            return None
        return [projected_avgs.get(stat, 0.0) for stat in INTERMEDIATE_PARAMS]

    def get_stats_last_7(self, player):
        try:
            projected_avgs = player.stats['2023_last_7']['avg']
        except KeyError as ke:
            return None
        return [projected_avgs.get(stat, 0.0) for stat in INTERMEDIATE_PARAMS]

    def gen_stats_table(self, player_list: list, swaps: dict = {}, comp_last_7: bool = False) -> DataFrame:
        # room to account for games played (GP)
        df = DataFrame(columns=INTERMEDIATE_PARAMS)
        active_players = [player for player in player_list if not player.injured]
        for player in active_players:
            if use_last_7 or (comp_last_7 and player in swaps.values()):
                stats = self.get_stats_last_7(player) or self.get_stats_projected(player)
            else:
                stats = self.get_stats_projected(player) or self.get_stats_last_7(player)
            if not stats:
                raise KeyError(f'Last 7 and projected stats not found for {player.name}...')
            df.loc[player.name] = stats

        df.loc['Total'] = df.sum()
        df.loc['Total']['FT%'] = df.loc['Total']['FTM'] / df.loc['Total']['FTA']
        df.loc['Total']['FG%'] = df.loc['Total']['FGM'] / df.loc['Total']['FGA']
        return df.drop(columns=['FGA', 'FGM', 'FTA', 'FTM'])

    def gen_comp_matrix(self, df: DataFrame) -> DataFrame:
        matrix = df.copy()
        for stat in SCORING_PARAMS:
            ascending = stat == 'TO'
            matrix = matrix.sort_values(stat, ascending=ascending)
            matrix[stat] = range(1, len(df) + 1)
        return matrix

    def get_curr_outcomes(self, home_team: str, curr_comp_totals: DataFrame, curr_stds: DataFrame) -> int:
        curr_outcomes = {}
        curr_wins = 0
        for away in league.get_teams().keys():
            if away == home_team:
                continue
            curr_outcome_df = league.calculate_1v1(curr_comp_totals, curr_stds, [home_team, away])
            curr_outcomes[away] = league.evaluate_1v1_outcome(curr_outcome_df)
        curr_wins = sum([1 if outcome > 0 else 0 for outcome in curr_outcomes.values()])
        return curr_outcomes, curr_wins


    def gen_totals_matrix(self, changed_rosters: dict = {}, swaps: dict = {}, comp_last_7: bool = False) -> DataFrame:
        # compare all roster totals
        rosters = self.get_rosters()
        # allow for player swap sims
        for team, roster in changed_rosters.items():
            if team in rosters:
                rosters[team] = roster
        totals = DataFrame(columns=SCORING_PARAMS)
        try:
            for team, roster in rosters.items():
                df = self.gen_stats_table(player_list=roster, swaps=swaps, comp_last_7=comp_last_7)
                totals.loc[team] = df.loc['Total']
        except KeyError as ke:
            raise KeyError(f'Error while processing {team}: {ke}')

        std = totals.std()
        return totals, std

    ''' This made no sense, redo later
    def calculate_winning_odds(self, certainty: list) -> int:
        comps = len(certainty)
        majority = len(certainty)//2 + 1
        winning_outcomes = [comb(comps, i) for i in range(majority, comps + 1)]
        #print(winning_outcomes)

    def odds_of_k_wins(self, certainty: list, k: int = 5, index: int = 0) -> int:
        res = 0
        trials = len(certainty)
        outcomes = comb(trials, k)
        for i in range(index, trials):
            #TODO: figure out how to calculate this somewhat efficiently
            j = 5
        #return res + odds_of_k_wins(certainty, k, index + 1)
        return 0
    '''

    def calculate_certainty(self, avg1, avg2, stat_std):
        # find the overlapping area of two normal distributions
        x = (avg1 + avg2)/2
        if avg1 > avg2:
            area = norm.cdf(x, avg1, stat_std) + (1. - norm.cdf(x, avg2, stat_std))
        else:
            area = norm.cdf(x, avg2, stat_std) + (1. - norm.cdf(x, avg1, stat_std))
        return area

    def calculate_1v1(self, comp_matrix: DataFrame, std, names: tuple) -> DataFrame:
        if len(names) != 2:
            raise Exception('Provide 2 teams to calculate outcome.')
        for name in names:
            if name not in comp_matrix.index:
                raise Exception('Invalid team provided.')
        if names[0] == names[1]:
            return None

        results_matrix = DataFrame(comp_matrix.columns)
        results_list = []
        certainty_list = []
        for stat in SCORING_PARAMS:
            avg_home = comp_matrix[stat][names[0]]
            avg_away = comp_matrix[stat][names[1]]
            if stat == 'TO':
                result = avg_home - avg_away
                results_list.append('W' if result < 0 else 'L')
            else:
                result = avg_home - avg_away
                results_list.append('W' if result > 0 else 'L')

            # calculate probability of overlap (draw or reversal)
            certainty = self.calculate_certainty(avg_home, avg_away, std[stat])
            # square for odds that both teams fall in overlapping range
            # take complement for probability that no overlap is possible (outcome certain)
            certainty = 1 - certainty * certainty
            certainty_list.append(certainty)

        results_matrix['Outcome'] = results_list
        results_matrix['Certainty'] = certainty_list

        # prep data for display
        display_df = comp_matrix.loc[[names[0], names[1]]]
        results_matrix = results_matrix.set_index(0).transpose()
        display_df = concat([display_df, results_matrix])
        return display_df

    def gen_std_matrix(self, comp_totals: DataFrame, stat_stds: DataFrame) -> DataFrame:
        std_matrix = (comp_totals - comp_totals.mean()) / stat_stds
        return std_matrix

    def find_free_agent(self, free_agents: list, name: str) -> list:
        possibles = []
        for agent in free_agents:
            if name in agent.name:
                possibles.append(agent)
        return possibles
        raise Exception('Invalid player name given.')

    def swap_sim(self, roster: list, prospect, name: str) -> list:
        for i in range(len(roster)):
            if name in roster[i].name:
                roster[i] = prospect
                return roster
        raise Exception(f'Swap failed. (choose someone valid to replace; given {name})')

    def simulate_swap(self, home: str, swaps: dict, free_agents: list, comp_last_7: bool = False) -> DataFrame:
        roster = league.get_team(home).roster.copy()
        for k, v in swaps.items():
            if not isinstance(v, str):
                v = v.name
            prospect = league.find_free_agent(free_agents, v)
            if len(prospect) != 1:
                raise Exception(f'Must select a free agent to simulate signing. Inputs: {v}, Output: {prospect}')
            roster = league.swap_sim(roster, prospect[0], k)
        new_roster = {home: roster}
        return league.gen_totals_matrix(new_roster, swaps, comp_last_7)

    def evaluate_1v1_outcome(self, outcome_df: DataFrame) -> int:
        won_certainties = [outcome_df.loc['Certainty'][i] for i in range(outcome_df.shape[1])
                           if outcome_df.loc['Outcome'][i] == 'W']
        lost_certainties = [outcome_df.loc['Certainty'][i] for i in range(outcome_df.shape[1])
                            if outcome_df.loc['Outcome'][i] == 'L']
        prioritized_losses = []

        if len(won_certainties) >= (outcome_df.shape[1]//2 + 1):
            return sum(won_certainties) / len(won_certainties)
        for i in range((outcome_df.shape[1]//2 + 1) - len(won_certainties)):
            min_certainty = 1.0
            for j in range(len(lost_certainties)):
                if lost_certainties[j] < min_certainty:
                    min_certainty = lost_certainties[j]
                    lost_certainties[j] = 1.0
            prioritized_losses.append(min_certainty)
        return -sum(prioritized_losses) / len(prioritized_losses)

    def get_swap_rank_delta(self, home: str, curr_totals: DataFrame, comp_matrix: DataFrame, swap_comp_matrix: DataFrame) -> DataFrame:
        printout = concat([curr_totals.loc[home], comp_matrix.loc[home], swap_comp_matrix.loc[home]], axis=1)
        printout.columns = ['Total', 'Current Rank', 'Swapped']
        return printout

    def swap_worthwhile(self, swap_totals: DataFrame, swap_stds: DataFrame, curr_outcomes: dict,
                        home: str, target: str = ''):
        worthwhile_tally = 0
        win_tally = 0
        # determine if swap is beneficial
        for away in league.get_teams().keys():
            if away == home or (target and target != away):
                continue
            # calculate wins and losses
            swapped_outcome_df = league.calculate_1v1(swap_totals, swap_stds, [home, away])
            outcome = league.evaluate_1v1_outcome(swapped_outcome_df)
            worthwhile = outcome - curr_outcomes[away]
            win_tally += 1 if outcome > 0 else 0
            worthwhile_tally += 1 if worthwhile > 0 else 0
            if target and target == away:
                return outcome, worthwhile
        return worthwhile_tally, win_tally

    def find_replacements_vs_all(self, to_replace: str, free_agents: list, curr_comp_totals: DataFrame,
                                 curr_stds: DataFrame, home: str, other_swaps: dict = {}, pos: list = []):
        results = {}
        start_time = time()
        curr_outcomes, curr_wins = self.get_curr_outcomes(home, curr_comp_totals, curr_stds)
        print('Finding optimal pickups for all matchups...')
        print('Starting player search...')
        for agent in free_agents:
            if pos:
                compatible = False
                for p in pos:
                    if p in agent.eligibleSlots:
                        compatible = True
                        break
                if not compatible:
                    continue
            other_swaps.update({to_replace: agent})
            try:
                swap_totals, swap_stat_stds = league.simulate_swap(home,
                                                                   other_swaps,
                                                                   free_agents, comp_last_7=True)
            except KeyError:
                print(f'Skipping {agent.name}...')
                continue

            # rank totals
            tally = league.swap_worthwhile(swap_totals, swap_stat_stds, curr_outcomes, home)
            if (tally[0] >= 4 and tally[1] >= curr_wins) or tally[1] > curr_wins:
                results[agent.name] = {'wins': tally[1], 'improvements': tally[0]}
                # swap_matrix = league.gen_comp_matrix(swap_totals)
                # print(f'\n{agent.name} outcomes...')
                # print(league.get_swap_rank_delta(comp_totals, comp_matrix, swap_matrix).to_string())

        elapsed_time = time() - start_time
        print(f'DONE: Found {len(results)} candidates in {elapsed_time}s')
        return {k: v for k, v in reversed(sorted(results.items(), key=lambda item: item[1]['wins']))} \
            if results else {}

    def find_replacements_vs_one(self, to_replace: str, free_agents: list, curr_comp_totals: DataFrame,
                                 curr_stds: DataFrame, home: str, away: str, other_swaps: dict = {}, pos: list = []):
        results = {}
        if away == home:
            raise Exception('Cannot replace: provide a valid opponent...')

        # find 1v1 outcome with current roster
        curr_outcome_df = league.calculate_1v1(curr_comp_totals, curr_stds, [home, away])
        curr_outcomes = {away: league.evaluate_1v1_outcome(curr_outcome_df)}

        start_time = time()
        print('Finding optimal pickups for 1v1...')
        print('Starting player search...')
        for agent in free_agents:
            if pos:
                compatible = False
                for p in pos:
                    if p in agent.eligibleSlots:
                        compatible = True
                        break
                if not compatible:
                    continue
            other_swaps.update({to_replace: agent})
            try:
                swap_totals, swap_stat_stds = league.simulate_swap(home,
                                                                   other_swaps,
                                                                   free_agents, comp_last_7=True)
            except KeyError:
                print(f'Skipping {agent.name}...')
                continue

            # rank totals
            tally = league.swap_worthwhile(swap_totals, swap_stat_stds, curr_outcomes, home, away)
            if tally[1] > 0:
                results[agent.name] = {'avg_ctnty': tally[0], 'swap_eval': tally[1]}
                # swap_matrix = league.gen_comp_matrix(swap_totals)
                # print(f'\n{agent.name} outcomes...')
                # print(league.get_swap_rank_delta(comp_totals, comp_matrix, swap_matrix).to_string())

        elapsed_time = time() - start_time
        print(f'DONE: Found {len(results)} candidates in {elapsed_time}s')
        return {k: v for k, v in reversed(sorted(results.items(), key=lambda item: item[1]['avg_ctnty']))} \
            if results else {}


# TODO: import schedule and predict outcomes by who's playing
# TODO: import team abbreviation/name dictionary to use schedule
# TODO: allow scoping optimizations by current match-up
if __name__ == '__main__':
    # get user input
    user = sys.argv[1] if len(sys.argv) > 1 else 'Team Bustos'
    opp = sys.argv[2] if len(sys.argv) > 2 else 'Team Panda'

    # construct league object and start espn connection
    league = FantasyAnalytics(league_id=LEAGUE_ID, year=YEAR,
                              espn_s2=ESPN_S2, swid=SWID, home=user)

    # consolidate totals
    comp_totals, stat_stds = league.gen_totals_matrix()
    std_matrix = league.gen_std_matrix(comp_totals, stat_stds)

    # rank totals
    comp_matrix = league.gen_comp_matrix(comp_totals)
    curr_outcomes, curr_wins = league.get_curr_outcomes(user, comp_totals, stat_stds)
    print(f'Current win total: {curr_wins}\nCurrent outcome ratings: {curr_outcomes}')

    # and once more to find difference in ranking
    agents = league.free_agents(size=100)
    #print(league.find_free_agent(free_agents=agents, name='Claxton')[0].proTeam)

    optimal_gen = league.find_replacements_vs_all('Conley', agents, comp_totals, stat_stds, user,
                                                  other_swaps={}, pos=['SF', 'PF'])
    print(optimal_gen)
    optimal_scoped = league.find_replacements_vs_one('Conley', agents, comp_totals, stat_stds, user, opp,
                                                     other_swaps={}, pos=['SF', 'PF'])
    print(optimal_scoped)



#Grant Williams, Dennis Smith for McDaniels, Conley

