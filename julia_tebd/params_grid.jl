# Edit this grid to change what gets scanned. t, U, Vpp are fixed here since
# only N and Vpm are being scanned right now; add them as loops too if needed.
const Ns        = [128, 256]
const Vpms      = [4.0, 4.35]
const operators = ["cup", "Sz"]   # Sz for benchmark, cdn, cup for phi at long dist
const t, U, Vpp = 1.0, 0.1, 0.8

grid() = [(N, t, U, Vpp, Vpm, op) for N in Ns for Vpm in Vpms for op in operators]
