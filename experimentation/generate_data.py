from simulator import Simulator
import pandas as pd
import os


def generate_data_csv(
    path, days=365 * 3, n_wells=50, n_technicians=5, seed=17, datadrift_example=False,
):

    print("generating data")
    sim = Simulator(n_wells=n_wells, n_technicians=n_technicians, seed=seed,)

    # get one year's worth:
    times = []
    wells = [[] for _ in range(n_wells)]
    simdays = 28 if datadrift_example else days
    for d in range(simdays):
        print("simulating data for day", d, end="\r")
        for h in range(24):
            sim.step_hour()
            state = sim.get_state()
            times.append(state["time"])
            for well in state["wells"]:
                wells[well["id"]].append(well)

    # for each well, get time to failure etc.
    print()
    first = True
    for well in wells:
        print("processing well", well[0]["id"], end="\r")
        df = pd.DataFrame(well)
        df["time"] = times

        df = df.sort_values(["id", "time"])

        df = df[['id', 'time', 'state', 'issue', 'temperature', 'pressure', 'load', 'production_rate']]

        # and write:
        if not datadrift_example:
            df.to_csv(path, index=False, header=first, mode="w" if first else "a")
        else:
            duration = 7 * 4  # 4 weeks
            drift_at = 7 * 3  # 3 weeks
            n = 0
            for (date, date_df) in df.groupby(df.time.dt.date):
                filename = os.path.join(path, date.strftime("%Y%m%d") + ".csv")
                if n > drift_at:
                    date_df = date_df.copy()
                    date_df["temperature"] *= 10
                date_df.to_csv(filename, index=False, header=first, mode="w" if first else "a")
                n += 1
                # just do 4 weeks
                if n >= duration:
                    break
        first = False


if __name__ == "__main__":
    from pathlib import Path

    fpath = Path(__file__).absolute().parent.parent / "data" / "oilwells.csv"
    print("saving data to", fpath)
    generate_data_csv(fpath)
    fpath = Path(__file__).absolute().parent.parent / "data" / "per_day"
    os.makedirs(fpath, exist_ok=True)
    # generate_data_csv(fpath, datadrift_example=True)
