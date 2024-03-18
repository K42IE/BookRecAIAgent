# Book Recommender Agent

This is an AI book recommender agent that helps booksellers recommend in-stock books.

## Installing and Running 

1. You'll need [Milvus](https://milvus.io/) for the vector database.

2. We recommend using Docker with Milvus:
https://milvus.io/docs/install_standalone-docker.md
https://hub.docker.com/

3. In your root folder you will need to add an file named ".env". This is where you will put your OpenAI API key in the format: 

OPENAI_API_KEY=_your openai api key goes here_

Keep in mind that you will need to go to the OpenAI website and create a key using an account. Usage of they key may cost money, please read OpenAI's policy (currently there is a free trial to acuess their APIs).

## Helpful Links
UI: https://www.gradio.app/

## Data Notes
!! Be aware that this code uses all files in the "data" directory. This is expected to be used to hold information about the invetory in a bookstore. You can change how the program inputs code by editing the "main.py" file where we are setting the SimpleDirectoryReader (currently line 42 (the meaning of life lol (if you get the reference ^_^)).

Original Sample Database (oldDataset_books.csv): https://www.kaggle.com/datasets/arpansri/books-summary?resource=download

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

