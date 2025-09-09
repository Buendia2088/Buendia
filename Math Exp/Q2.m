x = linspace(0, 10, 200);

f1 = @(x) 1 ./ x;
f2 = @(x) x .^ 2;
f3 = @(x) exp(-x);
f4 = @(x) 1 ./ (x.^2 + 1);
f5 = @(x) x .* sin(x);

y1 = f1(x);
y2 = f2(x);
y3 = f3(x);
y4 = f4(x);
y5 = f5(x);

figure;

subplot(5, 1, 1);
for i = 1:length(x)
    plot(x(1:i), y1(1:i), 'b');
    title('f(x) = 1/x as x \rightarrow \infty');
    xlabel('x');
    ylabel('f(x)');
    ylim([0, 1]);
    grid on;
    drawnow;
end

pause(1);

subplot(5, 1, 2);
for i = 1:length(x)
    plot(x(1:i), y2(1:i), 'r');
    title('f(x) = x^2 as x \rightarrow \infty');
    xlabel('x');
    ylabel('f(x)');
    ylim([0, max(y2)]);
    grid on;
    drawnow;
end

pause(1);

subplot(5, 1, 3);
for i = 1:length(x)
    plot(x(1:i), y3(1:i), 'g');
    title('f(x) = e^{-x} as x \rightarrow \infty');
    xlabel('x');
    ylabel('f(x)');
    ylim([0, 1]);
    grid on;
    drawnow;
end

subplot(5, 1, 4);
for i = 1:length(x)
    plot(x(1:i), y4(1:i), 'b');
    title('f(x) = 1/(x^2 + 1) as x \rightarrow \infty');
    xlabel('x');
    ylabel('f(x)');
    ylim([0, 1]);
    grid on;
    drawnow;
end

pause(1);

subplot(5, 1, 5);
for i = 1:length(x)
    plot(x(1:i), y5(1:i), 'r');
    title('f(x) = x * sin(x) as x \rightarrow \infty');
    xlabel('x');
    ylabel('f(x)');
    ylim([-max(abs(y5)), max(abs(y5))]);
    grid on;
    drawnow;
end
