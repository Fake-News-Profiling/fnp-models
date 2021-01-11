import sys, argparse
from twython import Twython
from reporter.reporter import ReportBuilder

NUM_TWEETS = 100 # Profile a user using their last 100 tweets

# Twitter API Handler
class TwitterHandler():
    def __init__(self, num_tweets_to_get: int, api_key: str, api_secret: str):
        self.num_tweets_to_get = num_tweets_to_get
        self.twitter = Twython(api_key, api_secret)
    
    def user_timeline(self, username: str):
        """ Get the users last num_tweets_to_get tweets from their tweet feed, returning 
        them as a list of tweets """
        timeline = self.twitter.get_user_timeline(
            screen_name=username, count=400, tweet_mode='extended')
        
        def extract_tweet_contents(tweet):
            return {
                'text': tweet['full_text'],
                'id': tweet['id']
            }
        
        timeline_contents = list(
            filter(lambda a: len(a['text']) > 10, 
            [extract_tweet_contents(tweet) for tweet in timeline])) # Get tweets with more than 10 characters
        print(f'Found {len(timeline_contents)} usable tweets (from successfully fetching {len(timeline_contents)}/199 tweets):\n{timeline_contents}')
        return timeline_contents[:self.num_tweets_to_get]

def parse_program_args():
    """ Parse the inputted program arguments """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--api_key', '-k', help='The Twitter API key', type=str)
    arg_parser.add_argument('--api_secret', '-s', help='The Twitter API secret', type=str)
    arg_parser.add_argument('--classifier_weights', '-w', 
        help='Filepath to the classifiers weights checkpoint - a file ending in .ckpt')
    return vars(arg_parser.parse_args())

def main():
    # Parse command-line args, and create a TwitterHandler and ReportBuilder object
    args = parse_program_args()
    if not (args['api_key'] and args['api_secret'] and args['classifier_weights']):
        raise Exception("Invalid command-line arguments. Use '--help' to see which " 
            + "arguments the program should be run with.")

    twitter_handler = TwitterHandler(NUM_TWEETS, args['api_key'], args['api_secret'])
    report_builder = ReportBuilder(args['classifier_weights'])

    # Start taking user input
    print("Starting up the program. Type 'exit' at any time to stop.")
    while True:
        username = input("Enter the username of the user to profile: ")
        if username == 'exit':
            break

        try: 
            timeline = twitter_handler.user_timeline(username)
            report_builder.generate_report(username, timeline)
        except Exception as err:
            print(f"An error occurred: {err}")

if __name__ == '__main__':
    main()
