function [Vps] = VPS(A, B, Z, h, f, Cx, Cy, theta)

den1 = (-A*B^2*f^2*h*cos(theta)*sin(theta)^2-A*B^2*f^2*h*cos(theta)^3);

vp1x = (-(A*B^2*f^3*h*sin(theta)^3)-(A*B^2*Cx*f^2*h*cos(theta)*sin(theta)^2)-(A*B^2*f^3*h*cos(theta)^2*sin(theta))-(A*B^2*Cx*f^2*h*cos(theta)^3))/den1;

vp1y = (-(A*B^2*Cy*f^2*h*cos(theta)*sin(theta)^2)-(A*B^2*Cy*f^2*h*cos(theta)^3))/den1;


den2 = (-A^2*B*f^2*h*sin(theta)^3-A^2*B*f^2*h*cos(theta)^2*sin(theta));

 
vp2x = (-(A^2*B*Cx*f^2*h*sin(theta)^3)+(A^2*B*f^3*h*cos(theta)*sin(theta)^2)-(A^2*B*Cx*f^2*h*cos(theta)^2*sin(theta))+(A^2*B*f^3*h*cos(theta)^3))/den2;
vp2y = (-(A^2*B*Cy*f^2*h*sin(theta)^3)-(A^2*B*Cy*f^2*h*cos(theta)^2*sin(theta)))/den2;

Vps = [vp1x,vp1y;vp2x, vp2y];
end
