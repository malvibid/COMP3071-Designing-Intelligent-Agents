import random
import tkinter as tk


class PoemBlackboard:
    def __init__(self, agents, update_interval, poem):
        """
        Initializes a new PoemBlackboard object.
        Args:

            agents: A list of agents that can be executed on the lines of the poem.
            update_interval: The interval (in milliseconds) between updates to the visualization.
            poem: A list of strings representing the lines of the poem.
        """
        self.coordi_x = 100
        self.coordi_padding_y = 100
        self.update_interval = update_interval
        self.agents = agents
        self.lines = self.generate_lines(poem)
        self.create_visualization()

    def generate_lines(self, poem):
        """
        Generate a list of lines for a poem, along with their coordinates and tags.

        Args:
            poem (list): A list of strings representing the lines of the poem.

        Returns:
            list: A list of tuples representing each line in the poem. Each tuple contains the x and y
            coordinates of the line, the line itself, and a tag in the format "line{line_number}".
        """
        lines = []
        for idx, line in enumerate(poem):
            y = self.coordi_padding_y  + idx*50
            tag = f"line{idx}"
            lines.append([self.coordi_x, y, line, tag])
        return lines

    def create_visualization(self):
        """
        Create the visualization window, canvas, and draw the lines on the canvas.

        Returns:
            None
        """
        # Create the main window for the visualization
        self.window = tk.Tk()

        # Set the window to a fixed size and disallow resizing
        self.window.resizable(False, False)

        # Create the canvas to draw the visualization on
        self.canvas = tk.Canvas(self.window, width=1000, height=1000)

        # Pack the canvas so it fills the entire window
        self.canvas.pack()

        # Draw each line on the canvas
        for line in self.lines:
            # Use the x and y coordinates from the line tuple to position the text
            # The anchor parameter is set to "W" to left-align the text
            self.canvas.create_text(line[0], line[1], text=line[2], tags=line[3], anchor=tk.W)

    def run(self):
        """
        Executes a randomly chosen agent on the lines, updates the visualization, and schedules the next update.

        Returns:
            None
        """

        # Get a list of the names of the available agents
        agent_names = [agent.__name__ for agent in self.agents]

        # Choose a random agent from the list of available agents
        agent = random.choice(self.agents)

        # Print the name of the chosen agent for debugging purposes
        print(f"Executing {agent.__name__} out of {agent_names}")

        # Apply the chosen agent to the lines to modify them
        self.lines = agent(self.lines)

        # Clear the canvas and draw the updated lines
        self.canvas.delete("all")
        for line in self.lines:
            self.canvas.create_text(line[0], line[1], text=line[2], tags=line[3], anchor=tk.W)

        # Schedule the next update to occur after the specified interval
        self.canvas.after(self.update_interval, self.run)

    def start(self):
        """
        Starts the main loop for the visualization.
        """
        # Call the run method to update the visualization and schedule the next update
        self.run()

        # Start the main loop for the visualization window
        self.window.mainloop()


if '__main__' == "__main__":
    from lab7.sol.agent_ls import remove_adjective,replace_with_synonym

    # lines from a poem by Dorianne Laux
    txt = [
        "Once there was a dog and a cat",
        "Who decided to have an extended natter",
        "It's the best part of the day, morning light sliding",
        "down rooftops, treetops, the birds pulling themselves",
        "up out of whatever stupor darkened their wings",
        "night still in their throats",
    ]


    agent_cons = [remove_adjective, replace_with_synonym]
    timer_update = 1000


    # Create a new PoemBlackboard object
    bb = PoemBlackboard(agent_cons, timer_update, txt)
    # Start the visualization
    bb.start()

