targetNumber = randi([1, 100]);
maxAttempts = 7;
attempts = 0;
fprintf('Guess the number between 1 and 100. You have %d attempts.\n', maxAttempts);

while attempts < maxAttempts
    attempts = attempts + 1;
    guess = input(sprintf('Attempt %d: Enter your guess: ', attempts));
    if guess > targetNumber
        fprintf('High\n');
    elseif guess < targetNumber
        fprintf('Low\n');
    else
        fprintf('You won!\n');
        break;
    end
    if attempts == maxAttempts
        fprintf('Sorry, you have used all your attempts. The number was %d.\n', targetNumber);
    end
end
