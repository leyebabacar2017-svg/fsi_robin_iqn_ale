SetFactory("OpenCASCADE");

ls = 0.35;
hs = 0.02;

Rectangle(1) = {0.25,0.19,0,ls,hs};

Physical Surface("Beam") = {1};
