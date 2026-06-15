SetFactory("OpenCASCADE");

xc = 0.20;
yc = 0.20;
R  = 0.05;

Disk(1) = {xc,yc,0,R,R};

Physical Surface("Cylinder") = {1};
