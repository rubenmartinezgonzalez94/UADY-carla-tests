%inftyLine.m

figure(1);
clf;
plot([0,1920,1920,0,0],[0,0,1080,1080,0],'w');
axis([0,1920,0,1080],"equal");
hold on

vp = zeros(3,2);

[a,b] = ginput(1);
a=round(a);
b=round(b);
vp(:,1)=[a;b;1];
plot(vp(1,1),vp(2,1),"*r");

[a,b] = ginput(1);
a=round(a);
b=round(b);
vp(:,2)=[a;b;1];
plot(vp(1,2),vp(2,2),"*r");


vl=cross(vp(:,1),vp(:,2));
vl=vl/vl(3);

x=[0,1920];
y=[-(vl(3)+vl(1)*x)/vl(2)];
plot([x(1),x(2)], [y(1),y(2)],'c');


p=zeros(3,3);

[a,b] = ginput(1);
a=round(a);
b=round(b);
p(:,1)=[a;b;1];
plot(p(1,1),p(2,1),"*k");

[a,b] = ginput(1);
a=round(a);
b=round(b);
p(:,2)=[a;b;1];
plot(p(1,2),p(2,2),"*k");

[a,b] = ginput(1);
a=round(a);
b=round(b);
p(:,3)=[a;b;1];
plot(p(1,3),p(2,3),"*k");

l=zeros(3,4);
l(:,1) =cross(vp(:,2), p(:,1))';
l(:,2) =cross(vp(:,2), p(:,2))';
l(:,3) =cross(vp(:,1), p(:,3))';
l(:,4) =cross(vp(:,1), p(:,2))';

c=zeros(3,4);
c(:,1) = cross(l(:,1), l(:,4));
c(:,2) = cross(l(:,1), l(:,3));
c(:,3) = cross(l(:,2), l(:,3));
c(:,4) = cross(l(:,4), l(:,2));

for i=1:4
	c(:,i) /= c(3,i);
	plot(c(1,i),c(2,i),'om');
end
pc=[c,c(:,1)];
plot(pc(1,:),pc(2,:),"b");

if c(:,4)==p(:,2)
	display("Todo esta bien");
end

H=eye(3);
H(3,:)=vl';

pcp=H*pc;

for i = 1:5
	pcp(:,i) /= pcp(3,i);
end
figure(2);
clf
plot(pcp(1,:),pcp(2,:),"b");
axis equal;






