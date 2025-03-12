function [Vps] = VPS(f, Cx, Cy, theta)

vp1x = f * tan(theta) + Cx;
vp1y = Cy;

vp2x = Cx - f / tan(theta);
vp2y = Cy;

Vps = [vp1x, vp1y; vp2x, vp2y];
end
