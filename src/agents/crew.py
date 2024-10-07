from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from agents.tools.search import SearchTools

@CrewBase
class AgentsCrew():
    """Agents crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def market_researcher(self) -> Agent:
        config = self.agents_config['market_researcher']
        llm_config = self.agents_config.get('llm')
        
        if not llm_config:
            raise ValueError("LLM configuration is missing")

        model_name = llm_config['model']

        llm = LLM(
            model=model_name,
            temperature=llm_config['temperature'],
            base_url=llm_config['base_url'],
            api_key=llm_config['api_key']
        )
        
        return Agent(
            config=config,
            llm=llm,
            tools=[
              SearchTools.search_internet,
              SearchTools.search_instagram,
              SearchTools.open_page,
            ],
            verbose=True,
        )

    @agent
    def content_strategist(self) -> Agent:
        config = self.agents_config['content_strategist']
        llm_config = self.agents_config.get('llm')

        llm = None
        if llm_config:
            llm = LLM(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                base_url=llm_config['base_url'],
                api_key=llm_config['api_key']
            )

        return Agent(
            config=config,
            llm=llm,
            verbose=True,
        )

    @agent
    def visual_creator(self) -> Agent:
        config = self.agents_config['visual_creator']
        llm_config = self.agents_config.get('llm')

        llm = None
        if llm_config:
            llm = LLM(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                base_url=llm_config['base_url'],
                api_key=llm_config['api_key']
            )

        return Agent(
            config=config,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def copywriter(self) -> Agent:
        config = self.agents_config['copywriter']
        llm_config = self.agents_config.get('llm')

        llm = None
        if llm_config:
            llm = LLM(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                base_url=llm_config['base_url'],
                api_key=llm_config['api_key']
            )

        return Agent(
            config=config,
            llm=llm,
            verbose=True,
        )

    @task
    def market_research(self) -> Task:
        return Task(
            config=self.tasks_config["market_research"],
            agent=self.market_researcher(),
            output_file="market_research.md",
        )

    @task
    def content_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["content_strategy"],
            agent=self.content_strategist(),
        )

    @task
    def visual_content_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config["visual_content_creation"],
            agent=self.visual_creator(),
            output_file="visual-content.md",
        )

    @task
    def copywriting_task(self) -> Task:
        return Task(
            config=self.tasks_config["copywriting"],
            agent=self.copywriter(),
        )

    @task
    def report_final_content_strategy(self) -> Task:
        return Task(
            config=self.tasks_config["report_final_content_strategy"],
            agent=self.content_strategist(),
            output_file="final-content-strategy.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
