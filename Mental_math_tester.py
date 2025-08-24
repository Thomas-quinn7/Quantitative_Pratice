import random 
import time
start_time = time.time()
score = 0
given_ans="No"
user_ans=0
game_over = False
"""
    Just small programme to pratice mental maths

    operation_sel will randomise each time:

        1 = multiple
        2 = division
        3 = adding
        4 = subtracting

        change the numbers of a/b random.randint() to change the range of the numbers
        asked to operate on
        random.randint(a,b) generates number N
        a <= N <= b

        1.5 seconds of delay added to timer to be fair to play
        """

while given_ans!="done" and not game_over:
    if score == 0:
        print("To end the game please type  'done'")
    operation_sel=random.randint(1,5)
    if operation_sel==1:
        a = random.randint(12,100)
        b = random.randint(12,100)
        answer = a*b
        while answer != user_ans:
            given_ans = input(f"{a}*{b}=")
            if given_ans == "done":
                game_over = True
                break
            try:
                user_ans = int(given_ans)
            except ValueError:
                print("Please enter a number or 'done'")
                continue
        
        if not game_over:
            score += 1
            print("Correct!")
    
    elif operation_sel==2:
        a = random.randint(12,100)
        b = random.randint(12,100)
        Main_num = a*b
        answer = a
        while answer != user_ans:
            given_ans = input(f"{Main_num}/{b}=")
            if given_ans == "done":
                game_over = True
                break
            try:
                user_ans = int(given_ans)
            except ValueError:
                print("Please enter a number or 'done'")
                continue
        
        if not game_over:
            score += 1
            print("Correct!")

    elif operation_sel==3:
        a = random.randint(100,1000)
        b = random.randint(100,1000)
        answer = a+b
        while answer != user_ans:
            given_ans = input(f"{a}+{b}=")
            if given_ans == "done":
                game_over = True
                break
            try:
                user_ans = int(given_ans)
            except ValueError:
                print("Please enter a number or 'done'")
                continue
        
        if not game_over:
            score += 1
            print("Correct!")
    
    elif operation_sel==4:
        a = random.randint(100,1000)
        b = random.randint(100,1000)
        if a >= b:
            answer = a-b
            while answer != user_ans:
                given_ans = input(f"{a}-{b}=")
                if given_ans == "done":
                    game_over = True
                    break
                try:
                    user_ans = int(given_ans)
                except ValueError:
                    print("Please enter a number or 'done'")
                    continue
        else:
            answer = b-a
            while answer != user_ans:
                given_ans = input(f"{b}-{a}=")
                if given_ans == "done":
                    game_over = True
                    break
                try:
                    user_ans = int(given_ans)
                except ValueError:
                    print("Please enter a number or 'done'")
                    continue

        if not game_over:
            score += 1
            print("Correct!")

end_time = time.time()
elapsed = end_time - start_time + 1.5
print("=" * 100)
print("Thanks for playing")
print(f"Final score: {score}")
print(f"Time taken: {elapsed:.2f} seconds")