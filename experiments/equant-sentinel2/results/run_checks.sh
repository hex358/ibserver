set -x
python3 scripts/inject_test.py --method glrt --amps 100,200,400 --len-px 2 --wid-px 2 --dm 0
python3 scripts/inject_test.py --method excl --amps 100,200,400 --len-px 2 --wid-px 2 --dm 0
python3 scripts/run_events.py --obj STARLINK-31550 --orbits gp_starlink.tle --start 2026-11-19T18:00 --end 2026-11-19T18:30 --dm 10 --out results/v3
echo ALLDONE
