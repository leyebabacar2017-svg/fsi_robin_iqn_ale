SetFactory("OpenCASCADE");

xc = 0.20;
yc = 0.20;
R  = 0.05;

ls = 0.35;
hs = 0.02;

// Cylindre
Disk(1) = {xc,yc,0,R,R};

// Drapeau
Rectangle(2) = {0.25,0.19,0,ls,hs};

Physical Surface("Cylinder") = {1};
Physical Surface("Beam")     = {2};
