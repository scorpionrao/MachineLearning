function deckList = loadDeckList()
%GETDECKLIST reads the fixed deck list in deck.txt and returns a
%cell array of the words
%   deckList = GETDECKLIST() reads the fixed deck list in deck.txt 
%   and returns a cell array of the words in deckList.


%% Read the fixed movieulary list
fid = fopen('deck_ids.txt');

% Store all decks in cell array deck{}
n = 45;  % Total number of decks 

deckList = cell(n, 1);
for i = 1:n
    % Read line
    line = fgets(fid);
    % Word Index (can ignore since it will be = i)
    [idx, deckName] = strtok(line, ' ');
    % Actual Word
    deckList{i} = strtrim(deckName);
end
fclose(fid);

end
