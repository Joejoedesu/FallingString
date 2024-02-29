/** 
 * StarterCode for "Attack of the Blobs!" 
 * CS248B Fundamentals of Computer Graphics: Animation & Simulation
 * 
 * Fill in the the missing code (see TODO items).
 * Try reducing MAX_BLOBS to 1 to get started. 
 * Good luck!!
 * 
 * @author Doug L. James <djames@cs.stanford.edu> 
 * @date 10/28/2022
 */

const MAX_BLOBS = 100; /// TODO: 100 or more to complete "Attack of the Blobs!" challenge. Use just a few for testing. 
const DRAW_BLOB_PARTICLES = true;

const STIFFNESS_STRETCH = 1000.0;
const STIFFNESS_BEND = 2.6;
const STIFFNESS_AREA = 1.0;
const STRETCH_DAMP = 4.5;

const WIDTH = 1024;
const HEIGHT = 1024;
const PARTICLE_RADIUS = WIDTH / 400.0; // for rendering
const PARTICLE_MASS = 1.0;
const BLOB_PARTICLES = 12;
const BLOB_RADIUS = WIDTH/20; //WIDTH / 20;
const d0 = 2*PARTICLE_RADIUS;

const COLOR_fill = [[215,201,212], [200,221,196], [207,193,180], [226,218,191], [153,190,198], [217,227,234], [232,216,215]];
const COLOR_out = [[193,175,183],[170,193,165],[165,145,143], [213,193,157], [122,158,169], [166,194,207], [216,183,180]];

const Gridsubdiv = 2;
const dx = WIDTH/Gridsubdiv;
const dy = HEIGHT/Gridsubdiv;
let frameRate = 60;
let ItemsInGrid = [];

//////// IMPORTANT ARRAYS OF THINGS /////////
let particles = []; // All particles in the scene (rigid + blobs)
let edges = []; //     All edges in the scene (rigid + blobs)
let blobs = []; //     All blobs in the scene (increases over time)
let environment; //    Environment with all rigid edges available as getEdges()

let isPaused = true;
let nTimesteps = 0; // #frame-length timesteps taken
let detectedEdgeEdgeFailure = false; // Halts simulation and turns purple if true -- blobs win!

// Graph paper texture map:
let bgImage;

function preload() {
	bgImage = loadImage('dango-wallpaper.jpg');
}

function setup() {
	createCanvas(WIDTH, HEIGHT);
	background(100);
	ellipseMode(RADIUS);
	environment = new Environment();
	//print("|particles|=" + particles.length + ",  |edge|=" + edges.length + ",  |blobs|=" + blobs.length);
}

// Computes moving-average FPS
function updateFrameRate() {
	let frameRateNow = 1000.0 / deltaTime; //instantaneous estimate
	let alpha = min(0.5, 3.0 / frameRateNow); // short-time moving average for low FPS
	if (deltaTime > 0) frameRate = (1 - alpha) * frameRate + alpha * frameRateNow; // moving-average FPS
}

/// Timesteps (w/ substeps) and draws everything.
function draw() {

	///// SIMULATE /////
	if (!isPaused) {
		if (nTimesteps % 10 == 0) {
			if (blobs.length < MAX_BLOBS)
				createRandomBlob();
		}

		let dtFrame = 0.01;
		let nSubsteps = 1; // #times to split dtFrame
		for (let step = 0; step < nSubsteps; step++)
			advanceTime(dtFrame / nSubsteps);
		nTimesteps++;
	}

	///// RENDER /////
	push();
	background(0);
	environment.draw();
	for (let blob of blobs)
		blob.draw();
	pop();
	drawMouseForce();

	/// TEXT OUTPUT:
	push();
	textSize(18);
	noStroke();
	fill(0);
	updateFrameRate();
	text("#BLOBS: " + blobs.length, 10, 20);
	text("#EDGES: " + edges.length, 10, 40);
	text("#PARTICLES: " + particles.length, 10, 60);
	text("#FRAME RATE: " + round(frameRate*10)/10, 10, 80);
	pop();
}

function keyPressed() {
	if (keyCode == 32) // spacebar
		isPaused = !isPaused;
}

function getColliPair(){
	let bb = [];
	let eb = [];
	for (let grid of ItemsInGrid){
		let t_eb = findeb_pair(grid[0], grid[1]);
		let t_bb = findbb_pair(grid[1]);
		bb = bb.concat(t_bb);
		eb = eb.concat(t_eb);
	}
	return [eb, bb];
}

function advanceTime(dt) {
	environment.advanceTime(dt);
	
	for(let blob of blobs){
		blob.updateAABB(dt);
	}
	
	// getItemsInGrid();
		
	// let eb = findeb_pair(environment.eAABBpair, blobs);
	// let bb = findbb_pair(blobs);
	let eb = [];
	let bb = [];
	if (ItemsInGrid.length == 0){
		eb = findeb_pair(environment.eAABBpair, blobs);
	  bb = findbb_pair(blobs);
	} else{
		let temp = getColliPair();
		eb = temp[0];
		bb = temp[1];
	}
	
	//////////////////////////////////////
	////// GATHER PARTICLE FORCES ////////
	{
		// Clear forces:
		for (let particle of particles)
			particle.f.set(0, 0);

		gatherParticleForces_Gravity();

		// Damping (springs or otherwise -- you can add some if you want): 

		// Blob springs: 
		for (let blob of blobs) {
			blob.gatherForces_Stretch();
			blob.gatherForces_Bend();
			blob.gatherForces_Area();
		}

		gatherParticleForces_Penalty(eb, bb);

		// Mouse force (modify if you want):
		applyMouseForce();
	}

	//////////////////////////////////////////
	// Update velocity (using mass filtering):
	for (let particle of particles)
		acc(particle.v, dt * particle.invMass(), particle.f)

	//////////////////////////////////////////
	// Collision filter: Correct velocities //
	applyPointEdgeCollisionFilter(dt, eb, bb);

	//////////////////////////////////////////
	// Update positions:
	for (let particle of particles)
		acc(particle.p, dt, particle.v)
	
	getItemsInGrid();
	verifyNoEdgeEdgeOverlap();
}

function AABBinside(point, AABB){
	return (point.x >= AABB[0].x && point.x <= AABB[1].x && point.y >= AABB[0].y && point.y <= AABB[1].y);
}

function AABB_AABB(AABB1, AABB2){
	let p1 = AABB1[0];
	let p2 = AABB1[1];
	let p3 = vec2(p1.x, p2.y);
	let p4 = vec2(p2.x, p1.y);
	let ch1 = AABBinside(p1, AABB2) || AABBinside(p2, AABB2) || AABBinside(p3, AABB2) || AABBinside(p4, AABB2);
	let q1 = AABB2[0];
	let q2 = AABB2[1];
	let q3 = vec2(q1.x, q2.y);
	let q4 = vec2(q2.x, q1.y);
	let ch2 = AABBinside(q1, AABB1) || AABBinside(q2, AABB1) || AABBinside(q3, AABB1) || AABBinside(q4, AABB1);
	return ch1 || ch2;
}

function findeb_pair(edgeGroup, blobGroup){
	let eb_pair = [];
	for (let eAABB of environment.eAABBpair){
		for (let blob of blobs){
			if (AABB_AABB(eAABB[0], blob.AABB)){
				eb_pair.push([eAABB[1], blob]);
			}
		}
	}
	return eb_pair;
}

function findbb_pair(blobGroup){
	let bb_pair = [];
	for (let i = 0; i < blobGroup.length - 1; i++){
		for (let j = i+1; j < blobGroup.length; j++){
			if (AABB_AABB(blobGroup[i].AABB, blobGroup[j].AABB)){
				bb_pair.push([blobGroup[i], blobGroup[j]]);
			}
		}
	}
	return bb_pair;
}
			
function PointEdgeCollision(edge, particle, dt, e){
	let isColli = false;
	
	let a = sub(edge.r.p, edge.q.p);
	let a_d = sub(edge.r.v, edge.q.v);
	let b = sub(particle.p, edge.q.p);
	let b_d = sub(particle.v, edge.q.v);

	let cur = vec2(edge.r.v.x - edge.q.v.x, edge.r.v.y - edge.q.v.y);
	let cur_n = vec2(cur.x/cur.mag(), cur.y/cur.mag());
	let cur_m = abs(dot(a, cur_n))/a.mag();

	let A = a_d.x*b_d.y-a_d.y*b_d.x;
	let B = a_d.x*b.y-a_d.y*b.x + a.x*b_d.y-a.y*b_d.x;
	let C = a.x*b.y-a.y*b.x;
	let D = B*B - 4*A*C;
	let t = -1.0;
	let mark = 0;

	if (A != 0 && D >= 0){
		//print("CASE1");
		mark = 1;
		let r = -1/2*(B+B/abs(B)*sqrt(D));
		let t1 = r/A;
		let t2 = C/r;
		let tmin = min(t1, t2);
		let tmax = max(t1, t2);
		if (tmin >= 0) t = tmin;
		else t = tmax;
	}
	else if (A == 0 && B != 0){
		// print("CASE2");
		if (-C/B >= 0 && ((-C/B < t && t >= 0) || t < 0)){
			mark = 2;
			t = -C/B; 
		}
	}
	if (t > 0 && t <= dt){
		let ps = vec2(particle.p.x + particle.v.x*t, particle.p.y + particle.v.y*t);
		let q_ = vec2(edge.q.p.x + edge.q.v.x * t, edge.q.p.y + edge.q.v.y*t);
		let r_ = vec2(edge.r.p.x + edge.r.v.x * t, edge.r.p.y + edge.r.v.y*t);
		let ps_q = vec2(ps.x-q_.x,ps.y-q_.y);
		let ps_q_norm = vec2(ps_q.x/ps_q.mag(),ps_q.y/ps_q.mag());
		a = vec2(r_.x-q_.x, r_.y-q_.y);
		let alp = dot(ps_q_norm, a);
		let alp_n = ps_q.mag()/a.mag()*alp/abs(alp);
		if (alp_n >= 0 && alp_n <= 1){
			isColli = true;
			let alp_n_inv = 1 - alp_n;
			let meff = 1/(particle.invMass() + alp_n_inv*alp_n_inv*edge.q.invMass() + alp_n*alp_n*edge.r.invMass());
			let c = vec2(edge.q.v.x + alp_n*(edge.r.v.x - edge.q.v.x), edge.q.v.y + alp_n*(edge.r.v.y - edge.q.v.y));
			let neg_v = vec2(c.x - particle.v.x, c.y - particle.v.y);

			let n0 = vec2(q_.y-r_.y, r_.x - q_.x);
			let n0_norm = vec2(n0.x/n0.mag(), n0.y/n0.mag());
			let sign = dot(n0_norm, neg_v)/abs(dot(n0_norm, neg_v));
			let n = vec2(n0_norm.x*sign, n0_norm.y*sign);
			let gamma = (1+e)*dot(neg_v, n)*meff;
			particle.v = add(particle.v, vec2(gamma*n.x*particle.invMass(), gamma*n.y*particle.invMass()));
			edge.q.v = sub(edge.q.v, vec2(alp_n_inv*gamma*n.x*edge.q.invMass(), alp_n_inv*gamma*n.y*edge.q.invMass()));
			edge.r.v = sub(edge.r.v, vec2(alp_n*gamma*n.x*edge.r.invMass(), alp_n*gamma*n.y*edge.r.invMass()));			
			// break Loop1;
		}
	}
	return isColli;
}

function applyPointEdgeCollisionFilter(dt, eb, bb) {
	// FIRST: Just rigid edges.
	let e = 0.4;
	let isColli = true;
	let max_it = 100;
	while (isColli && max_it > 0){
		// print("round");
		isColli = false;
		max_it -= 1;
		for (let eb_pair of eb){
			let edge_ = eb_pair[0];
			let blob_ = eb_pair[1];
			for (let particle of blob_.BP){
				isColli = isColli || PointEdgeCollision(edge_, particle, dt, e);
			}
			for (let edge of blob_.BE){
				isColli = isColli || PointEdgeCollision(edge, edge_.q, dt, e);
				isColli = isColli || PointEdgeCollision(edge, edge_.r, dt, e);
			}
		}

		// SECOND: All rigid + blob edges (once you get this ^^ working)
		for (let bb_pair of bb){
			let blob1 = bb_pair[0];
			let blob2 = bb_pair[1];
			for (let edge of blob1.BE){
				for (let particle of blob2.BP){
					isColli = isColli || PointEdgeCollision(edge, particle, dt, e);
				}
			}
			for (let edge of blob2.BE){
				for (let particle of blob1.BP){
					isColli = isColli || PointEdgeCollision(edge, particle, dt, e);
				}
			}
		}
	}
}

function getItemsInGrid(){
	ItemsInGrid = [];
	for (let i = 0; i < Gridsubdiv; i++){
		for (let j = 0; j < Gridsubdiv; j++){
			let GridRange = [vec2(i*dx, j*dy), vec2((i+1)*dx, (j+1)*dy)];
			let GridEdge = [];
			let GridBlob = [];
			
			//get environment
			for (let eAABB of environment.eAABBpair){
				if (AABB_AABB(eAABB[0], GridRange)){
					GridEdge.push(eAABB);
				}
			}
			//get blob
			for (let blob of blobs){
				if (AABB_AABB(blob.AABB, GridRange)){
					GridBlob.push(blob);
				}
			}
			ItemsInGrid.push([GridEdge, GridBlob]);
		}
	}
	// ItemsInGrid = [[environment.eAABBpair, blobs]];
}
			
// Efficiently checks that no pair of edges overlap, where the pairs do not share a particle in common.
function verifyNoEdgeEdgeOverlap() {
	if (detectedEdgeEdgeFailure) return; // already done
	
	// getItemsInGrid();
	for (let grid of ItemsInGrid){
		//check edges
		for (let edge of grid[0]){
			let ei = edge[1];
			for (let blob of grid[1]){
				for (let ej of blob.BE){
					if (checkEdgeEdgeOverlap(ei, ej)){
						let cur = vec2(ej.r.v.x - ej.q.v.x, ej.r.v.y - ej.q.v.y);
						let cur_n = vec2(cur.x/cur.mag(), cur.y/cur.mag());
						let cur_m = abs(dot(sub(ej.r.p, ej.q.p), cur_n))/sub(ej.r.p, ej.q.p).mag();
						print("edge1", ei);
						print("edge2", ej);
						print("curl", cur_m);
						detectedEdgeEdgeFailure = true;
						isPaused = true;
						return;
					}
				}
			}
		}
		//check blob
		for (let i = 0; i < grid[1].length - 1; i++){
			for (let j = i + 1; j < grid[1].length; j++){
				for (let ei of grid[1][i].BE){
					for (let ej of grid[1][j].BE){
						if (checkEdgeEdgeOverlap(ei, ej)){
							let cur = vec2(ej.r.v.x - ej.q.v.x, ej.r.v.y - ej.q.v.y);
							let cur_n = vec2(cur.x/cur.mag(), cur.y/cur.mag());
							let cur_m = abs(dot(sub(ej.r.p, ej.q.p), cur_n))/sub(ej.r.p, ej.q.p).mag();
							print("edge1", ei);
							print("edge2", ej);
							print("curl", cur_m);
							detectedEdgeEdgeFailure = true;
							isPaused = true;
							return;
						}
					}
				}
			}
		}
	}
}


function checkEdgeEdgeOverlap(ei, ej) {
	//let eps = -2.0e-7;
	let p1 = ei.q.p;
	let q1 = ei.r.p;
	let p2 = ej.q.p;
	let q2 = ej.r.p;
	let exp1 = tri_area(p1, q1, p2) * tri_area(p1, q1, q2);
	let exp2 = tri_area(p2, q2, p1) * tri_area(p2, q2, q1);
	if (exp1 < 0 && exp2 < 0){
		print(exp1, exp2);
	}
	return (exp1 < 0 && exp2 < 0);
}

function addPenalityToBlob(edge, blob, k, c){
	let qr = sub(edge.r.p, edge.q.p);
	let qr_norm = vec2(qr.x/qr.mag(), qr.y/qr.mag());

	for (let particle of blob.BP){
		let qp = sub(particle.p, edge.q.p);
		let alp = dot(qr_norm, qp);
		let n = sub(qp, vec2(alp*qr_norm.x, alp*qr_norm.y));
		let d = max(HEIGHT, WIDTH);

		if (alp >= 0 && alp <= qr.mag()){
			d = n.mag();

		} else if (sub(particle.p, edge.q.p).mag() <= d0){
			n = sub(particle.p, edge.q.p);
			d = n.mag();				
		} else if (sub(particle.p, edge.r.p).mag() <= d0){
			n = sub(particle.p, edge.r.p);
			d = n.mag();
		}
		if (d <= d0){
			let n_norm = vec2(n.x/n.mag(), n.y/n.mag());
			let f = vec2(k*(d0-d)*n_norm.x, k*(d0-d)*n_norm.y);
			let f_dam = vec2(-c*dot(particle.v, n_norm)*n_norm.x, -c*dot(particle.v, n_norm)*n_norm.y);
			f = add(f, f_dam);
			particle.f = add(particle.f, vec2(f.x/Gridsubdiv, f.y/Gridsubdiv));
		}
	}
}
	

// Computes penalty forces between all point-edge pairs
function gatherParticleForces_Penalty(eb, bb) {
	let k = 8000.0;
	let c = 10.0;

	for (let eb_pair of eb) {
		let edge = eb_pair[0];
		let blob = eb_pair[1];
		addPenalityToBlob(edge, blob, k/2, c);
	}

	for (let bb_pair of bb){
		let blob1 = bb_pair[0];
		let blob2 = bb_pair[1];
		for (let edge of blob1.BE){
			addPenalityToBlob(edge, blob2, k, c);
		}
		for (let edge of blob2.BE){
			addPenalityToBlob(edge, blob1, k, c);
		}
	}
}

function gatherParticleForces_Gravity() {
	let g = vec2(0, 100); //grav accel
	for (let particle of particles)
		acc(particle.f, particle.mass, g); // f += m g
}

// Blob currently being dragged by mouse forces, or undefined.
let mouseBlob;
// Selects closest blob for mouse forcing (mask out if using a GUI)
function mousePressed() {
	if (blobs.length == 0 || isPaused) return;

	// Set mouseBlob to blob closest to the mouse:
	let m = vec2(mouseX, mouseY);
	let minDist = 1000000000;
	let minCOM;
	let minBlob;
	for (let blob of blobs) {
		let com = blob.centerOfMass();
		if (com.dist(m) < minDist) {
			minDist = com.dist(m);
			minBlob = blob;
			minCOM = com;
		}
	}
	mouseBlob = minBlob;
}

function mouseReleased() {
	mouseBlob = undefined;
}

// Applies spring + damping force to all mouseBlob particles
function applyMouseForce() {
	if (mouseIsPressed && mouseBlob) {
		if (blobs.length < 1) return;
		let m = vec2(mouseX, mouseY);
		let blobCOM = mouseBlob.centerOfMass();
		let blobDist = blobCOM.dist(m);
		let mforce = sub(m, blobCOM).normalize().mult(100 * clamp(blobDist, 0, 100));


		// Apply force to blob particles:
		let P = mouseBlob.blobParticles();
		for (let part of P) {
			part.f.add(mforce);
			acc(part.f, -10.0, part.v); //some damping
		}
	}
}

// Draws line from the mouse to any forced mouseBlob
function drawMouseForce() {
	if (mouseIsPressed && mouseBlob) {
		if (blobs.length < 1) return;
		let m = vec2(mouseX, mouseY);
		let blobCOM = mouseBlob.centerOfMass();
		push();
		stroke(0);
		strokeWeight(5);
		line(m.x, m.y, blobCOM.x, blobCOM.y);
		pop();
	}
}

function tri_area(p0, p1, p2){
	let A_t = p0.x*p1.y-p0.y*p1.x + p1.x*p2.y-p1.y*p2.x + p2.x*p0.y-p2.y*p0.x;
	return A_t/2;
}

// Creates a default particle and adds it to particles list
function createParticle(x, y) {
	let p = new Particle(vec2(x, y), 1.0, PARTICLE_RADIUS);
	particles.push(p);
	return p;
}

class Particle {
	constructor(pRest, mass, radius) {
		this.pRest = vec2(pRest.x, pRest.y);
		this.p = vec2(pRest.x, pRest.y);
		this.v = vec2(0, 0);
		this.pin = false; // true if doesn't respond to forces
		this.mass = mass;
		this.radius = radius;
		this.f = vec2(0, 0);
	}
	invMass() {
		return (this.pin ? 0.0 : 1.0 / this.mass);
	}
	// Emits a circle
	draw() {
		// nobody = (this.pin ? fill("red") : fill(0)); // default colors (red if pinned)
		circle(this.p.x, this.p.y, this.radius); //ellipseMode(RADIUS);
	}
}

// Creates edge and adds to edge list
function createEdge(particle0, particle1) {
	let edge = new Edge(particle0, particle1);
	edges.push(edge);
	return edge;
}

// Edge spring
class Edge {
	// Creates edge spring of default stiffness, STIFFNESS_STRETCH
	constructor(particle0, particle1) {
		this.q = particle0;
		this.r = particle1;
		this.restLength = this.q.pRest.dist(this.r.pRest);
		this.stiffness = STIFFNESS_STRETCH;
	}
	// True if both particles are pinned
	isRigid() {
		return (this.q.pin && this.r.pin);
	}
	// Current length of edge spring
	length() {
		return this.q.p.dist(this.r.p);
	}
	// Rest length of edge spring
	lengthRest() {
		return this.restLength;
	}
	// Draws the unstylized line 
	draw() {
		let a = this.q.p;
		let b = this.r.p;
		line(a.x, a.y, b.x, b.y);
	}
}

// RIGID ENVIRONMENT COMPOSED OF LINE SEGMENTS (pinned Edges)
class Environment {

	constructor() {
		this.envParticles = [];
		this.envEdges = [];
		this.eAABBpair = [];

		///// BOX /////
		let r = PARTICLE_RADIUS;
		this.p00 = createParticle(r, r);
		this.p01 = createParticle(r, HEIGHT - r);
		this.p11 = createParticle(WIDTH - r, HEIGHT - r);
		this.p10 = createParticle(WIDTH - r, r);
		this.p00.pin = this.p01.pin = this.p11.pin = this.p10.pin = true;
		this.envParticles.push(this.p00);
		this.envParticles.push(this.p01);
		this.envParticles.push(this.p11);
		this.envParticles.push(this.p10);
		this.envEdges.push(createEdge(this.p00, this.p01));
		this.envEdges.push(createEdge(this.p01, this.p11));
		this.envEdges.push(createEdge(this.p11, this.p10));
		this.envEdges.push(createEdge(this.p10, this.p00));

		///// OBSTACLES FOR FUN (!) /////
		for (let i = 0.5; i < 4; i++) {
			this.createEnvEdge(i * width / 5, height / 2, (i + 1) * width / 5, height * 0.75);
		}
		this.updateAABBpair();
	}

	// Returns all rigid-environment edges.
	getEdges() {
		return this.envEdges;
	}

	// Creates a rigid edge.
	createEnvEdge(x0, y0, x1, y1) {
		let p0 = createParticle(x0, y0);
		let p1 = createParticle(x1, y1);
		p0.pin = true;
		p1.pin = true;
		let e = createEdge(p0, p1);
		this.envParticles.push(p0);
		this.envParticles.push(p1);
		this.envEdges.push(e);
	}

	// Updates any moveable rigid elements
	advanceTime(dt) {}
	
	updateAABBpair(){
		for (let edge of this.envEdges){
			let minx = min(edge.q.p.x, edge.r.p.x);
			let maxx = max(edge.q.p.x, edge.r.p.x);
			let miny = min(edge.q.p.y, edge.r.p.y);
			let maxy = max(edge.q.p.y, edge.r.p.y);
			let eAABB = [vec2(minx-3*d0,miny-3*d0), vec2(maxx+3*d0,maxy+3*d0)];
			this.eAABBpair.push([eAABB, edge]);
		}
	}

	// Makes popcorn <jk> no it doesn't... 
	draw() {
		push();
		image(bgImage, 0, 0, WIDTH, HEIGHT);

		if (detectedEdgeEdgeFailure) { // HALT ON OVERLAP + DRAW PURPLE SCREEN
			push();
			fill(191, 64, 191, 150);
			rect(0, 0, width, height);
			pop();
		}

		stroke("black");
		strokeWeight(PARTICLE_RADIUS);
		for (let edge of this.envEdges) {
			edge.draw();
		}
		fill("black");
		noStroke();
		for (let particle of this.envParticles) {
			particle.draw();
		}
		pop(); // wait, it does pop :/ 
	}
}

// Creates a blob centered at (x,y), and adds things to lists (blobs, edges, particles).
function createBlob(x, y) {
	let b = new Blob(vec2(x, y));
	blobs.push(b);
	return b;
}

// Tries to create a new blob at the top of the screen. 
function createRandomBlob() {
	for (let attempt = 0; attempt < 5; attempt++) {
		let center = vec2(random(2 * BLOB_RADIUS, WIDTH - 2 * BLOB_RADIUS), BLOB_RADIUS * 1.3); //random horizontal spot
		// CHECK TO SEE IF NO BLOBS NEARBY:
		let tooClose = false;
		for (let blob of blobs) {
			let com = blob.centerOfMass();
			if (com.dist(center) < 3 * blob.radius) // too close
				tooClose = true;
		}
		// if we got here, then center is safe:
		if (!tooClose) {
			createBlob(center.x, center.y);
			return;
		}
	}
}

class Blob {
	constructor(centerRest) {
		this.radius = BLOB_RADIUS;
		this.centerRest = centerRest; // original location

		// CREATE PARTICLES:
		this.BP = []; //blob particles
		this.n = BLOB_PARTICLES;
		let v0 = vec2(random(-100, 100), random(200, 220));
		for (let i = 0; i < this.n; i++) {
			let xi = this.radius * cos(i / this.n * TWO_PI) + centerRest.x;
			let yi = this.radius * sin(i / this.n * TWO_PI) + centerRest.y;
			let particle = createParticle(xi, yi);
			particle.v.set(v0);
			this.BP.push(particle);
		}
		
		this.AABB = [vec2(0,0), vec2(0,0)];
		this.updateAABB(0);
		
		this.area = this.computeArea();

		// CREATE EDGES FOR STRETCH SPRINGS + COLLISIONS:
		this.BE = []; // blob edges
		for (let i = 0; i < this.n; i++) {
			let p0 = this.BP[i];
			let p1 = this.BP[(i + 1) % this.n];
			this.BE.push(createEdge(p0, p1));
		}
		
		//let dc = 26;
		//this.fillColor = color([221 + random(-dc, dc), 160 + random(-dc, dc), 221 + random(-dc, dc), 255]); // ("Plum"); // 221, 160, 221
		//this.fillcolor = color([221, 160, 221, 255]);
		let dc = 0;
		let c = floor(random(0, COLOR_fill.length));
		this.fillColor = color([COLOR_fill[c][0], COLOR_fill[c][1], COLOR_fill[c][2], 255]);
		this.outline = color([COLOR_out[c][0], COLOR_out[c][1], COLOR_out[c][2], 255]);
	}

	blobParticles() {
		return this.BP;
	}
	
	updateAABB(dt){
		let minx = this.BP[0].p.x;
		let miny = this.BP[0].p.y;
		let maxx = this.BP[0].p.x;
		let maxy = this.BP[0].p.y;
		for(let part of this.BP){
			if (min(part.p.x, part.p.x + dt*part.v.x) < minx) minx = min(part.p.x, part.p.x + dt*part.v.x);
			if (max(part.p.x, part.p.x + dt*part.v.x) > maxx) maxx = max(part.p.x, part.p.x + dt*part.v.x);
		 	if (min(part.p.y, part.p.y + dt*part.v.y) < miny) miny = min(part.p.y, part.p.y + dt*part.v.y);
			if (max(part.p.y, part.p.y + dt*part.v.y) > maxy) maxy = max(part.p.y, part.p.y + dt*part.v.y);
		}
		this.AABB = [vec2(minx-d0,miny-d0), vec2(maxx+d0,maxy+d0)];
	}

	// Loops over blob edges and accumulates stretch forces (Particle.f += ...)
	gatherForces_Stretch() {
		let k = STIFFNESS_STRETCH;
		let c = STRETCH_DAMP;
		let vom = this.velocityOfMassCenter();
		
		for (let edge of this.BE) {
			let str_f = k*(edge.length() - edge.lengthRest());
			let v1 = sub(edge.r.p, edge.q.p);
			let f1 = vec2(str_f*v1.x/v1.mag(), str_f*v1.y/v1.mag());
			let f2 = vec2(-f1.x, -f1.y);
			f1 = add(f1, vec2(-c*(edge.q.v.x - vom.x), -c*(edge.q.v.y - vom.y)));
			f2 = add(f2, vec2(-c*(edge.r.v.x - vom.x), -c*(edge.r.v.y - vom.y)));
			edge.q.f = add(edge.q.f, f1);
			edge.r.f = add(edge.r.f, f2);
		}
	}
	// Loops over blob particles and accumulates bending forces (Particle.f += ...)
	gatherForces_Bend() {
		let k = STIFFNESS_BEND;
		for (let i = 0; i < this.n; i++) {
			let p0 = this.BP[(i + this.n - 1) % this.n];
			let p1 = this.BP[i];
			let p2 = this.BP[(i + 1) % this.n];

			let a = sub(p1.p, p0.p);
			let b = sub(p2.p, p1.p);
			let dot_ab = dot(a,b);
			let f0 = sub(b, vec2(a.x*dot_ab, a.y*dot_ab));
			f0 = vec2(f0.x*(-k/(2*a.mag())), f0.y*(-k/(2*a.mag())));
			let f2 = sub(a, vec2(b.x*dot_ab, b.y*dot_ab));
			f2 = vec2(f2.x*k/(2*b.mag()), f2.y*k/(2*b.mag()));
			p1.f = vec2(p1.f.x - f0.x - f2.x, p1.f.y - f0.y - f2.y);
		}
	}
	// Loops over blob particles and gathers area compression forces (Particle.f += ...)
	gatherForces_Area() {
		let k = STIFFNESS_AREA;
		let coeff = -k*(this.computeArea() - this.area);
		
		for (let i = 0; i < this.n; i++) {
			let p0 = this.BP[(i + this.n - 1) % this.n];
			let p1 = this.BP[i];
			let p2 = this.BP[(i + 1) % this.n];

			let b = sub(p0.p, p2.p);
			let dA = vec2(-b.y/2, b.x/2);
			p1.f = vec2(p1.f.x + coeff*dA.x, p1.f.y + coeff*dA.y);
		}

	}

	// Center of mass of all blob particles
	centerOfMass() {
		let com = vec2(0, 0);
		for (let particle of this.BP)
			acc(com, 1 / this.BP.length, particle.p); // assumes equal mass
		return com;
	}
	
	velocityOfMassCenter() {
		let vom = vec2(0,0);
		let m = 0;
		for (let particle of this.BP) {
			m += particle.mass;
			acc(vom, particle.mass, particle.v);
		}
		return vec2(vom.x/m, vom.y/m);
	}

	// Center of velocity of all blob particles
	centerOfVelocity() {
		let cov = vec2(0, 0);
		for (let particle of this.BP)
			acc(cov, 1 / this.BP.length, particle.v); // assumes equal mass
		return cov;
	}
	
	computeArea() {
		let A = 0;
		let p0 = this.BP[0].p;
		for (let i = 2; i < this.n; i++){
			let p1 = this.BP[i-1].p;
			let p2 = this.BP[i].p;
			A += tri_area(p0, p1, p2);
		}
		return A;
	}
	
	// Something simple to keep rigid blobs inside the box:
	rigidBounceOnWalls() {
		let pos = this.centerOfMass();
		let vel = this.centerOfVelocity();

		let R = BLOB_RADIUS + PARTICLE_RADIUS;

		// Boundary reflection (only if outside domain AND still moving outward):
		if ((pos.x < R && vel.x < 0) ||
			(pos.x > width - R && vel.x > 0)) {
			for (let particle of this.BP)
				particle.v.x *= -0.4;
		}
		if ((pos.y < R && vel.y < 0) ||
			(pos.y > height - R && vel.y > 0)) {
			for (let particle of this.BP)
				particle.v.y *= -0.4;
		}
	}

	// Something simple to keep nonrigid blob particles inside the box:
	nonrigidBounceOnWalls() {
		let R = PARTICLE_RADIUS;
		for (let particle of this.BP) {
			let pos = particle.p;
			let vel = particle.v;

			// Boundary reflection (only if outside domain AND still moving outward):
			if ((pos.x < R && vel.x < 0) ||
				(pos.x > width - R && vel.x > 0)) {
				vel.x *= -0.4;
			}
			if ((pos.y < R && vel.y < 0) ||
				(pos.y > height - R && vel.y > 0)) {
				vel.y *= -0.4;
			}
		}
	}

	draw() {
		push();
		strokeWeight(PARTICLE_RADIUS);
		stroke(this.outline); //BlueViolet");
		fill(this.fillColor); { // draw blob
			beginShape(TESS);
			for (let particle of this.BP)
				vertex(particle.p.x, particle.p.y);
			endShape(CLOSE);
		}

		if (DRAW_BLOB_PARTICLES) {
			fill(this.outline);
			for (let particle of this.BP)
				circle(particle.p.x, particle.p.y, PARTICLE_RADIUS);
		}

		this.drawBlobFace();
		pop();
	}

	drawAngle(){
		let len = BLOB_RADIUS*0.2;
		push();
		strokeWeight(PARTICLE_RADIUS);
		stroke("black");
		{
			rectMode(CENTER);
			push();
			translate(-len/(2*sqrt(2)), -len/(2*sqrt(2)));
			rotate(PI/4);
			fill(0);
			rect(0, 0, len, 0.02*len);
			pop();
			push();
			translate(len/(2*sqrt(2)), -len/(2*sqrt(2)));
			rotate(-PI/4);
			fill(0);
			rect(0, 0, len, 0.02*len);
			pop();
		}
		pop();
	}
	
	drawBlobFace() {
		let com = this.centerOfMass();
		let other = this.BP[floor(this.BP.length/2)].p;
		if (this.BP.length % 2 != 0){
			other = vec2((this.BP[floor(this.BP.length/2)].p.x + this.BP[ceil(this.BP.length/2)].p.x)/2, (this.BP[floor(this.BP.length/2)].p.y + this.BP[ceil(this.BP.length/2)].p.y)/2);
		}
		let hori = sub(this.BP[0].p, other);
		let angle = acos(dot(hori, vec2(1, 0))/hori.mag());
		if (this.BP[0].p.y < other.y) angle = -angle;
		
		push();
		strokeWeight(PARTICLE_RADIUS);
		stroke("black");
		translate(com.x, com.y);
		rotate(angle);
		rectMode(CENTER);
		if(hori.mag() <= 2.2*BLOB_RADIUS){
			fill(0);
			rect(0, BLOB_RADIUS*0.25, BLOB_RADIUS*0.2, BLOB_RADIUS*0.01);
			rect(-BLOB_RADIUS*0.45, -BLOB_RADIUS*0.1, BLOB_RADIUS*0.06, BLOB_RADIUS*0.5);
			rect(BLOB_RADIUS*0.45, -BLOB_RADIUS*0.1, BLOB_RADIUS*0.06, BLOB_RADIUS*0.5);
			// fill(255);
			// rect(-BLOB_RADIUS*0.48, -BLOB_RADIUS*0.125, BLOB_RADIUS*0.1, BLOB_RADIUS*0.25);
			// rect(BLOB_RADIUS*0.42, -BLOB_RADIUS*0.125, BLOB_RADIUS*0.1, BLOB_RADIUS*0.25);
		} else{
			push();
			translate(BLOB_RADIUS*0.4, 0);
			rotate(PI/2);
			scale(1.5,1.5,1.5);
			this.drawAngle();
			pop();
			push();
			translate(-BLOB_RADIUS*0.4, 0);
			rotate(-PI/2);
			scale(1.5,1.5,1.5);
			this.drawAngle();
			pop();
			push();
			translate(0, BLOB_RADIUS*0.2);
			rotate(PI);
			this.drawAngle();
			pop();
		}
		//CENTER OF MASS eyeball for now :/
		// let com = this.centerOfMass();
		// let cov = this.centerOfVelocity();
		// stroke(0);
		// fill(255);
		// circle(com.x, com.y, 5 * PARTICLE_RADIUS);
		// fill(0);
		// circle(com.x + 0.01 * cov.x + 3 * sin(nTimesteps / 3), com.y + 0.01 * cov.y + random(-1, 1), PARTICLE_RADIUS);
		pop();
	}
}


/////////////////////////////////////////////////////////////////
// Some convenient GLSL-like macros for p5.Vector calculations //
/////////////////////////////////////////////////////////////////
function length(v) {
	return v.mag();
}

function dot(x, y) {
	return x.dot(y);
}

function dot2(x) {
	return x.dot(x);
}

function vec2(a, b) {
	return createVector(a, b);
}

function vec3(a, b, c) {
	return createVector(a, b, c);
}

function sign(n) {
	return Math.sign(n);
}

function clamp(n, low, high) {
	return constrain(n, low, high);
}

function add(v, w) {
	return p5.Vector.add(v, w);
}

function sub(v, w) {
	return p5.Vector.sub(v, w);
}

function absv2(v) {
	return vec2(Math.abs(v.x), Math.abs(v.y));
}

function maxv2(v, n) {
	return vec2(Math.max(v.x, n), Math.max(v.y, n));
}

function minv2(v, n) {
	return vec2(Math.min(v.x, n), Math.min(v.y, n));
}

function vertexv2(p) {
	vertex(p.x, p.y);
}

// v += a*w
function acc(v, a, w) {
	v.x += a * w.x;
	v.y += a * w.y;
}

function rotateVec2(v, thetaRad) {
	const c = cos(thetaRad);
	const s = sin(thetaRad);
	return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}
