pub mod segregation;
use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};
use std::usize;

/*
 * A particle in SOPS grid used in all the behavior's SOPS grids
 *  */
struct Particle {
    x: u8,
    y: u8,
    color: u8
}

/*
 * Main Class for the Aggregation Behaviour Experiments on SOPS grid
 * Defines how the genome is interpreted and how each transition of
 * particles is derived from the the genome. Also provides final SOPS
 * grid evaluations to assess the fitness score of the genome
 * NOTE: Extend/Refer this to create new Behaviour classes
 *  */
pub struct SOPSEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [u16; 6],
    sim_duration: u64,
    fitness_val: f64,
    size: usize,
    max_fitness: u64,
    arena_layers: u16,
    particle_layers: u16,
    phenotype_sum: u16
}


impl SOPSEnvironment {

    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn seed_rng(seed: u64) -> rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[inline]
    fn move_frng() -> fastrand::Rng {
        fastrand::Rng::new()
    }

    #[inline]
    fn grid_rng(size: usize) -> Uniform<usize> {
        Uniform::new(0, size)
    }

    #[inline]
    fn unfrm_move() -> Uniform<u64> {
        Uniform::<u64>::new(0, 1000)
    }

    #[inline]
    fn unfrm_dir() -> Uniform<usize> {
        Uniform::new(0, 6)
    }

    #[inline]
    fn directions() -> Vec<(i32, i32)> {
        vec![(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    /*
     * Initialize a SOPS grid and place particles based on particle layer and arena layer count
     * Parameters Particle layers and Arena layers refer to the complete hexagonal lattice layers
     * of the SOPS grid and this also defines the total density of particles in the arena.
     * Calculates Max edge count possible for all the particles irrespective of the color
     * Calculates Max edge count possible for all the particles of the same color
     * NOTE: Use the Same random Seed value to get the same random init config
     *  */
    pub fn init_sops_env(genome: &[u16; 6], arena_layers: u16, particle_layers: u16, seed: u64) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let num_particles = 6*particle_layers*(1+particle_layers)/2 + 1;
        let k = 3*particle_layers;
        let agg_edge_cnt: u64 = (k*(k+1)).into();
        let mut grid_rng = SOPSEnvironment::seed_rng(seed);
        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = 2;
                grid[(i+arena_layers+j) as usize][i as usize] = 2;
                j +=1;
            }
        }

        //init grid and particles
        while participants.len() < num_particles.into() {
            let i = grid_rng.sample(&SOPSEnvironment::grid_rng(grid_size));
            let j = grid_rng.sample(&SOPSEnvironment::grid_rng(grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    color: 0
                });
                grid[i][j] = 1;
            }
            
        }

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_fitness: agg_edge_cnt,
            arena_layers,
            particle_layers,
            phenotype_sum: genome.iter().sum()
        }
    }

    pub fn print_grid(&self) {
        println!("SOPS grid");
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                print!(" {} ", self.grid[i][j]);
            }
            println!("");
        }
    }

    /*
     * Func to calculate a particle's neighbor count
     *  */
    fn get_neighbors_cnt(&self, i: u8, j: u8) -> u8 {
        let mut cnt = 0;
        for idx in 0..6 {
            let new_i = (i as i32 + SOPSEnvironment::directions()[idx].0) as usize;
            let new_j = (j as i32 + SOPSEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                if self.grid[new_i][new_j] == 1 {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    /*
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let mut particle = &mut self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // Move particle if movement is within bound
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // check to see if move is valid ie. within bounds and not in an already occupied location
            if self.grid[new_i][new_j] == 1 || self.grid[new_i][new_j] == 2 {
                return false;
            } else {
                // move the particle
                self.grid[particle.x as usize][particle.y as usize] = 0;
                self.grid[new_i][new_j] = 1;
                particle.x = new_i as u8;
                particle.y = new_j as u8;
                return true;
            }
        } else {
            return false;
        }
    }

    /*
     * Func to move 'n' particles in random directions in the SOPS grid
     *  */
    fn move_particles(&mut self, cnt: usize) {
        let mut par_moves: Vec<(usize, (i32, i32))> = Vec::new();
        for _ in 0..cnt {
            // Choose a random particle for movement
            let par_idx = SOPSEnvironment::rng().sample(&self.unfrm_par());
            let particle: &Particle = &self.participants[par_idx];
            // Based on the neighbor count decide if movement is possible (not possible if 6 neighbors)
            let n_cnt = self.get_neighbors_cnt(particle.x, particle.y) as usize;
            if n_cnt == 6 {
                continue;
            }
            // Move basis probability given by the genome for moving for given 
            // #. of neighbors
            let move_prb: f64 =
                self.phenotype[n_cnt] as f64 / (self.phenotype_sum as f64);
            if SOPSEnvironment::move_frng().u64(1_u64..=10000)
                <= (move_prb * 10000.0) as u64
            {
                // Choose a direction at random (out of the 6)
                let move_dir = SOPSEnvironment::directions()
                    [SOPSEnvironment::rng().sample(&SOPSEnvironment::unfrm_dir())];
                par_moves.push((par_idx, move_dir));
            }
        }

        // Parallel execution
        /*
        let par_moves: Vec<(usize, (i32, i32))> = (0..cnt).into_par_iter().filter_map(|_| {
            let par_idx = SOPSEnvironment::move_frng().usize(0..self.participants.len());
            let particle: &Particle = &self.participants[par_idx];
            let n_cnt = self.get_neighbors_cnt(particle.x, particle.y) as usize;
            if n_cnt == 6 {
                return None;
            }
            let move_prb: f64 =
                self.phenotype[n_cnt] as f64 / (self.phenotype.iter().sum::<u16>() as f64);
            // if SOPSEnvironment::move_nrng().generate_range(1_u64..=1000)
            //     <= (move_prb * 1000.0) as u64
            // {
            //     let move_dir = SOPSEnvironment::directions()
            //         [SOPSEnvironment::move_nrng().generate_range(1..6)];
            //     par_moves.push((par_idx, move_dir));
            // }
            if SOPSEnvironment::move_frng().u64(1_u64..=1000)
                <= (move_prb * 1000.0) as u64
            {
                let move_dir = SOPSEnvironment::directions()
                    [SOPSEnvironment::move_frng().usize(1..6)];
                return Some((par_idx, move_dir));
            }
            return None;
        }).collect();
         */

        for moves in par_moves.iter() {
            self.move_particle_to(moves.0, moves.1);
        }
    }

    /*
     * Evaluate the measure of resultant configuration of SOPS grid
     * #. of total edges between every particle
     */
    pub fn evaluate_fitness(&self) -> u32 {
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            sum + self.get_neighbors_cnt(particle.x, particle.y) as u32
        });
        edges / 2
    }

    /*
     * Move a single particle at a time for 'sim_duration = fn(#. of particles)' times
     *  */
    pub fn simulate(&mut self, take_snaps: bool) -> u32 {
        for step in 0..self.sim_duration {
            self.move_particles((self.participants.len() as f32 * 0.03) as usize);
            if take_snaps && (step == (self.participants.len() as u64) || step == (self.participants.len() as u64).pow(2)) {
                self.print_grid();
                println!("Edge Count: {}", self.evaluate_fitness());
                println!("Fitness: {}", self.evaluate_fitness() as f32/ self.get_max_fitness() as f32);
            }
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness as f64;
        fitness
    }

    pub fn get_max_fitness(&self) -> u64 {
        self.max_fitness
    }

    pub fn get_participant_cnt(&self) -> usize {
        self.participants.len()
    }
}
