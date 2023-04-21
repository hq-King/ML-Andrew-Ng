X=zeros(1,100);
X(1,1)=1;
for i=1:100
    X(i+1)=X(i)-(X(i)^2-3*X(i)-exp(X(i)))/(2*X(i)-3-exp(X(i)));
    if X(i+1)-X(i)<0.000001
        break
    end
end
