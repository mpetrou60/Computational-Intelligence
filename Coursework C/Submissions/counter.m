load D2.mat

n1 = 0;
n2 = 0;
n3 = 0;
n4 = 0;
n5 = 0;

for i=1:length(Index)
    if Class(i) == 1
        n1 = n1 + 1;
    elseif Class(i) == 2
        n2 = n2 + 1;
    elseif Class(i) == 3
        n3 = n3 + 1;
    elseif Class(i) == 4
        n4 = n4 + 1;
    else
        n5 = n5 + 1;
    end
end




